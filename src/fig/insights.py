"""Insight generation for Financial Insight Generator.

This module converts numeric metrics into human-readable summaries.

Design goals:
- Simple, template-based text generation for now
- Keep functions modular so an LLM-based generator can later plug in
  behind similar interfaces (e.g. generate_overall_summary, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_currency(value: Optional[float]) -> str:
    """Format a numeric value as currency."""
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def _fmt_number(value: Optional[float], decimals: int = 2) -> str:
    """Format a generic number with thousand separators."""
    if value is None:
        return "N/A"
    return f"{value:,.{decimals}f}"


def _fmt_percent(value: Optional[float], decimals: int = 2) -> str:
    """Format a numeric value as percentage."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------------


def generate_overall_summary(overall: Dict[str, Any]) -> str:
    """Generate a high-level overview of the business performance."""
    date_min = overall.get("date_min")
    date_max = overall.get("date_max")

    date_range_str = "an unknown period"
    if date_min is not None and date_max is not None:
        try:
            date_range_str = f"{date_min.date()} to {date_max.date()}"
        except AttributeError:
            # if they are not Timestamp-like
            date_range_str = f"{date_min} to {date_max}"

    total_revenue = overall.get("total_revenue")
    total_cost = overall.get("total_cost")
    gross_profit = overall.get("gross_profit")
    gross_margin_pct = overall.get("gross_margin_pct")
    n_transactions = overall.get("n_transactions")
    avg_order_value = overall.get("avg_order_value")

    lines: List[str] = []
    lines.append(f"Overview for {date_range_str}:")
    lines.append(
        f"- Total revenue was {_fmt_currency(total_revenue)} "
        f"across {_fmt_number(n_transactions, 0)} transactions."
    )

    if avg_order_value is not None:
        lines.append(
            f"- The average order value was {_fmt_currency(avg_order_value)}."
        )

    if total_cost is not None and gross_profit is not None:
        lines.append(
            f"- Total cost was {_fmt_currency(total_cost)}, "
            f"resulting in gross profit of {_fmt_currency(gross_profit)}."
        )
        if gross_margin_pct is not None:
            lines.append(
                f"- Gross margin was {_fmt_percent(gross_margin_pct)}."
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Segment insights
# ---------------------------------------------------------------------------


def _describe_top_segment(
    seg_name: str,
    df_seg: pd.DataFrame,
    total_revenue: Optional[float],
    max_items: int = 3,
) -> str:
    """Generate a paragraph describing top segments for a given dimension."""
    if df_seg.empty:
        return f"No data available for {seg_name}."

    seg_label = seg_name.replace("_", " ").title()

    # We expect df_seg to have columns: seg_name, revenue, revenue_share_pct, order_count, ...
    cols = df_seg.columns

    if seg_name not in cols:
        # fallback: use first column name
        segment_col = cols[0]
    else:
        segment_col = seg_name

    top_df = df_seg.head(max_items)

    lines: List[str] = []
    lines.append(f"Top {seg_label}:")

    for _, row in top_df.iterrows():
        label = row[segment_col]
        revenue = float(row.get("revenue", 0.0))
        revenue_share_pct = row.get("revenue_share_pct")
        order_count = int(row.get("order_count", 0))

        share_str = (
            _fmt_percent(float(revenue_share_pct))
            if revenue_share_pct is not None
            else "N/A"
        )

        # If total_revenue is known, cross-check share
        if total_revenue and total_revenue > 0:
            computed_share = (revenue / total_revenue) * 100.0
            share_str = _fmt_percent(computed_share)

        lines.append(
            f"- {label}: {_fmt_currency(revenue)} across "
            f"{order_count} orders (approx. {share_str} of revenue)."
        )

    return "\n".join(lines)


def generate_segment_insights(
    segments: Dict[str, pd.DataFrame], overall: Dict[str, Any]
) -> str:
    """Generate insights for categories, products, customers, and channels."""
    total_revenue = overall.get("total_revenue")
    parts: List[str] = []

    ordered_segments = [
        ("category", "categories"),
        ("product", "products"),
        ("customer_id", "customers"),
        ("channel", "channels"),
    ]

    for seg_key, human_name in ordered_segments:
        if seg_key in segments:
            paragraph = _describe_top_segment(seg_key, segments[seg_key], total_revenue)
            parts.append(paragraph)

    if not parts:
        return "No segment-level insights available (no segment columns found)."

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Trend insights
# ---------------------------------------------------------------------------


def generate_trend_insights(trend: Dict[str, Any]) -> str:
    """Generate natural-language description of month-over-month revenue trend."""
    if not trend.get("has_enough_data", False):
        return "There is not enough data to compute a reliable month-over-month trend."

    current_period = trend.get("current_period")
    previous_period = trend.get("previous_period")
    current_rev = trend.get("current_revenue")
    previous_rev = trend.get("previous_revenue")
    abs_change = trend.get("absolute_change")
    pct_change = trend.get("percent_change")
    direction = trend.get("direction", "flat")

    dir_word = {
        "up": "increased",
        "down": "decreased",
        "flat": "remained roughly flat",
    }.get(direction, "changed")

    period_str = "the latest month"
    if current_period is not None and previous_period is not None:
        try:
            period_str = f"{previous_period.date()} to {current_period.date()}"
        except AttributeError:
            period_str = f"{previous_period} to {current_period}"

    if pct_change is None:
        pct_str = "N/A"
    else:
        pct_str = _fmt_percent(pct_change)

    lines = [
        f"Month-over-month trend ({period_str}):",
        f"- Revenue {dir_word} from {_fmt_currency(previous_rev)} "
        f"to {_fmt_currency(current_rev)}.",
    ]

    if abs_change is not None:
        change_direction = "an increase" if abs_change > 0 else "a decrease"
        if abs_change == 0:
            change_direction = "no change"
        lines.append(
            f"- This represents {change_direction} of "
            f"{_fmt_currency(abs_change)} ({pct_str})."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anomaly insights
# ---------------------------------------------------------------------------


def generate_anomaly_insights(anomaly: Dict[str, Any]) -> str:
    """Generate a brief description of last-day revenue anomaly status."""
    if not anomaly.get("has_enough_history", False):
        return "There is not enough recent daily history to perform anomaly detection."

    current_date = anomaly.get("current_date")
    current_rev = anomaly.get("current_revenue")
    mean = anomaly.get("history_mean")
    std = anomaly.get("history_std")
    z = anomaly.get("z_score")
    status = anomaly.get("status", "normal")

    try:
        date_str = current_date.date()
    except AttributeError:
        date_str = current_date

    lines: List[str] = []
    lines.append(f"Daily anomaly check for {date_str}:")

    lines.append(
        f"- Revenue was {_fmt_currency(current_rev)} compared to a "
        f"recent average of {_fmt_currency(mean)} (std dev {_fmt_currency(std)})."
    )

    if z is not None:
        lines.append(f"- The z-score for this day is {_fmt_number(z, 2)}.")

    if status == "high":
        lines.append(
            "- This day appears unusually strong compared to recent history."
        )
    elif status == "low":
        lines.append(
            "- This day appears unusually weak compared to recent history."
        )
    else:
        lines.append("- This day looks normal relative to recent history.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------


def generate_full_report(metrics_bundle: Dict[str, Any]) -> str:
    """Generate a full multi-section report from the metrics bundle.

    Sections:
    - Overview
    - Segment highlights
    - Trend
    - Anomaly check
    """
    overall = metrics_bundle.get("overall", {})
    segments = metrics_bundle.get("segments", {})
    monthly_trend = metrics_bundle.get("monthly_trend", {})
    anomaly = metrics_bundle.get("anomaly", {})

    overview_txt = generate_overall_summary(overall)
    segments_txt = generate_segment_insights(segments, overall)
    trend_txt = generate_trend_insights(monthly_trend)
    anomaly_txt = generate_anomaly_insights(anomaly)

    sections = [
        "FINANCIAL INSIGHT REPORT",
        "========================",
        "",
        "1. Overview",
        "-----------",
        overview_txt,
        "",
        "2. Segment Highlights",
        "---------------------",
        segments_txt,
        "",
        "3. Trend Analysis",
        "-----------------",
        trend_txt,
        "",
        "4. Daily Anomaly Check",
        "----------------------",
        anomaly_txt,
    ]

    return "\n".join(sections)
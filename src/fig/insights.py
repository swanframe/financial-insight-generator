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

from .i18n import get_translator


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_currency(value: Optional[float]) -> str:
    """Format a numeric value as currency."""
    if value is None:
        return "N/A"
    # Single-currency mode: Indonesian Rupiah (IDR)
    # Common display: no decimals, thousands separator "."
    s = f"{float(value):,.0f}".replace(",", ".")
    return f"Rp{s}"


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
# Overview
# ---------------------------------------------------------------------------


def generate_overall_summary(overall: Dict[str, Any], language: str = "en") -> str:
    """Generate a high-level overview of the business performance."""
    t = get_translator(language)

    date_min = overall.get("date_min")
    date_max = overall.get("date_max")

    if date_min is not None and date_max is not None:
        try:
            date_range_str = f"{date_min.date()} to {date_max.date()}"
        except AttributeError:
            # if they are not Timestamp-like
            date_range_str = f"{date_min} to {date_max}"
    else:
        date_range_str = t("report.overview.unknown_period")

    total_revenue = overall.get("total_revenue")
    total_cost = overall.get("total_cost")
    gross_profit = overall.get("gross_profit")
    gross_margin_pct = overall.get("gross_margin_pct")
    n_transactions = overall.get("n_transactions")
    avg_order_value = overall.get("avg_order_value")

    lines: List[str] = []
    lines.append(
        t("report.overview.heading", date_range=date_range_str)
    )
    lines.append(
        t(
            "report.overview.total_revenue",
            total_revenue=_fmt_currency(total_revenue),
            n_transactions=_fmt_number(n_transactions, 0),
        )
    )

    if avg_order_value is not None:
        lines.append(
            t(
                "report.overview.avg_order_value",
                avg_order_value=_fmt_currency(avg_order_value),
            )
        )

    if total_cost is not None and gross_profit is not None:
        lines.append(
            t(
                "report.overview.cost_and_profit",
                total_cost=_fmt_currency(total_cost),
                gross_profit=_fmt_currency(gross_profit),
            )
        )
        if gross_margin_pct is not None:
            lines.append(
                t(
                    "report.overview.gross_margin",
                    gross_margin_pct=_fmt_percent(gross_margin_pct),
                )
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Segment insights
# ---------------------------------------------------------------------------


def _describe_top_segment(
    seg_name: str,
    df_seg: pd.DataFrame,
    total_revenue: Optional[float],
    max_items: int,
    language: str,
) -> str:
    """Generate a paragraph describing top segments for a given dimension."""
    t = get_translator(language)

    if df_seg.empty:
        return t("report.segments.no_data_for_segment", segment_name=seg_name)

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
    lines.append(
        t("report.segments.top_segment_header", segment_label=seg_label)
    )

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
            t(
                "report.segments.item_line",
                label=label,
                revenue=_fmt_currency(revenue),
                order_count=order_count,
                share_str=share_str,
            )
        )

    return "\n".join(lines)


def generate_segment_insights(
    segments: Dict[str, pd.DataFrame],
    overall: Dict[str, Any],
    language: str = "en",
) -> str:
    """Generate insights for categories, products, customers, and channels."""
    t = get_translator(language)

    total_revenue = overall.get("total_revenue")
    parts: List[str] = []

    ordered_segments = [
        ("category", "categories"),
        ("product", "products"),
        ("customer_id", "customers"),
        ("channel", "channels"),
    ]

    for seg_key, _human_name in ordered_segments:
        if seg_key in segments:
            paragraph = _describe_top_segment(
                seg_key,
                segments[seg_key],
                total_revenue,
                max_items=3,
                language=language,
            )
            parts.append(paragraph)

    if not parts:
        return t("report.segments.no_segment_insights")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Trend insights
# ---------------------------------------------------------------------------


def generate_trend_insights(trend: Dict[str, Any], language: str = "en") -> str:
    """Generate natural-language description of month-over-month revenue trend."""
    t = get_translator(language)

    if not trend.get("has_enough_data", False):
        return t("report.trend.not_enough_data")

    current_period = trend.get("current_period")
    previous_period = trend.get("previous_period")
    current_rev = trend.get("current_revenue")
    previous_rev = trend.get("previous_revenue")
    abs_change = trend.get("absolute_change")
    pct_change = trend.get("percent_change")
    direction = trend.get("direction", "flat")

    dir_word_map = {
        "up": t("report.trend.direction_word.up"),
        "down": t("report.trend.direction_word.down"),
        "flat": t("report.trend.direction_word.flat"),
    }
    direction_word = dir_word_map.get(
        direction, t("report.trend.direction_word.changed")
    )

    if current_period is not None and previous_period is not None:
        try:
            prev_str = str(previous_period.date())
            curr_str = str(current_period.date())
        except AttributeError:
            prev_str = str(previous_period)
            curr_str = str(current_period)
        period_str = t(
            "report.trend.period_range",
            previous_period=prev_str,
            current_period=curr_str,
        )
    else:
        period_str = t("report.trend.period_latest_month")

    if pct_change is None:
        pct_str = "N/A"
    else:
        pct_str = _fmt_percent(pct_change)

    lines: List[str] = []
    lines.append(
        t("report.trend.heading", period_str=period_str)
    )
    lines.append(
        t(
            "report.trend.revenue_change_line",
            direction_word=direction_word,
            previous_revenue=_fmt_currency(previous_rev),
            current_revenue=_fmt_currency(current_rev),
        )
    )

    if abs_change is not None:
        if abs_change > 0:
            change_direction = t("report.trend.change_direction.increase")
        elif abs_change < 0:
            change_direction = t("report.trend.change_direction.decrease")
        else:
            change_direction = t("report.trend.change_direction.no_change")

        lines.append(
            t(
                "report.trend.change_line",
                change_direction=change_direction,
                absolute_change=_fmt_currency(abs_change),
                percent_change=pct_str,
            )
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anomaly insights
# ---------------------------------------------------------------------------


def generate_anomaly_insights(anomaly: Dict[str, Any], language: str = "en") -> str:
    """Generate a brief description of last-day revenue anomaly status."""
    t = get_translator(language)

    if not anomaly.get("has_enough_history", False):
        return t("report.anomaly.not_enough_history")

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

    date_str = str(date_str)

    lines: List[str] = []
    lines.append(
        t("report.anomaly.heading", date=date_str)
    )
    lines.append(
        t(
            "report.anomaly.line_revenue_vs_avg",
            current_revenue=_fmt_currency(current_rev),
            mean=_fmt_currency(mean),
            std=_fmt_currency(std),
        )
    )

    if z is not None:
        lines.append(
            t(
                "report.anomaly.line_zscore",
                z_score=_fmt_number(z, 2),
            )
        )

    if status == "high":
        lines.append(t("report.anomaly.status.high"))
    elif status == "low":
        lines.append(t("report.anomaly.status.low"))
    else:
        lines.append(t("report.anomaly.status.normal"))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------


def generate_full_report(
    metrics_bundle: Dict[str, Any],
    language: str = "en",
) -> str:
    """Generate a full multi-section report from the metrics bundle.

    Sections:
    - Overview
    - Segment highlights
    - Trend
    - Anomaly check
    """
    t = get_translator(language)

    overall = metrics_bundle.get("overall", {})
    segments = metrics_bundle.get("segments", {})
    monthly_trend = metrics_bundle.get("monthly_trend", {})
    anomaly = metrics_bundle.get("anomaly", {})

    overview_txt = generate_overall_summary(overall, language=language)
    segments_txt = generate_segment_insights(segments, overall, language=language)
    trend_txt = generate_trend_insights(monthly_trend, language=language)
    anomaly_txt = generate_anomaly_insights(anomaly, language=language)

    sections: List[str] = [
        t("report.title"),
        t("report.underline"),
        "",
        t("report.section.overview"),
        "-----------",
        overview_txt,
        "",
        t("report.section.segments"),
        "---------------------",
        segments_txt,
        "",
        t("report.section.trend"),
        "-----------------",
        trend_txt,
        "",
        t("report.section.anomaly"),
        "----------------------",
        anomaly_txt,
    ]

    return "\n".join(sections)
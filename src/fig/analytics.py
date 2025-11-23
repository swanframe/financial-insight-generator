"""Analytics and KPI computation for Financial Insight Generator.

Responsibilities:
- Compute overall metrics (revenue, cost, profit, margin, etc.)
- Compute time-based metrics (e.g. revenue by day/week/month)
- Compute segment metrics (categories, products, customers, channels)
- Compute simple trends (e.g. last month vs previous month)
- Detect basic revenue anomalies (e.g. last day vs recent average)

All functions are pure: they take DataFrames/config and return data structures
(dicts or DataFrames) without printing or doing I/O.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import Config


# ---------------------------------------------------------------------------
# Overall metrics
# ---------------------------------------------------------------------------


def compute_overall_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute overall summary metrics for the given transactions.

    Metrics include:
    - total_revenue
    - total_cost (if available)
    - gross_profit (if cost available)
    - gross_margin_pct (if cost available and revenue > 0)
    - n_transactions
    - avg_order_value
    - date_min, date_max

    Args:
        df: Cleaned transactions DataFrame with at least columns:
            - date (datetime)
            - amount (numeric)
            - optional: cost (numeric)

    Returns:
        dict with overall metrics.
    """
    if "amount" not in df.columns:
        raise ValueError("DataFrame must contain 'amount' column for metrics.")
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain 'date' column for metrics.")

    total_revenue = float(df["amount"].sum())
    total_cost: Optional[float] = None
    if "cost" in df.columns:
        total_cost = float(df["cost"].sum())

    gross_profit: Optional[float] = None
    gross_margin_pct: Optional[float] = None

    if total_cost is not None:
        gross_profit = total_revenue - total_cost
        if total_revenue > 0:
            gross_margin_pct = (gross_profit / total_revenue) * 100.0

    n_transactions = int(len(df))
    avg_order_value: Optional[float] = None
    if n_transactions > 0:
        avg_order_value = total_revenue / n_transactions

    date_min = df["date"].min()
    date_max = df["date"].max()

    return {
        "total_revenue": total_revenue,
        "total_cost": total_cost,
        "gross_profit": gross_profit,
        "gross_margin_pct": gross_margin_pct,
        "n_transactions": n_transactions,
        "avg_order_value": avg_order_value,
        "date_min": date_min,
        "date_max": date_max,
    }


# ---------------------------------------------------------------------------
# Time-series metrics
# ---------------------------------------------------------------------------


def _freq_from_granularity(granularity: str) -> str:
    """Map config.analytics.time_granularity to pandas offset alias."""
    granularity = granularity.lower()
    if granularity == "day":
        return "D"
    if granularity == "week":
        return "W"
    if granularity == "month":
        return "M"
    raise ValueError(
        f"Unsupported time granularity '{granularity}'. "
        "Expected one of: 'day', 'week', 'month'."
    )


def compute_revenue_time_series(
    df: pd.DataFrame, freq: str = "M"
) -> pd.DataFrame:
    """Compute revenue (and optional cost/profit) aggregated by time period.

    Args:
        df: Cleaned transactions DataFrame with 'date' and 'amount'.
        freq: Pandas offset alias, e.g. 'D', 'W', 'M'.

    Returns:
        DataFrame with columns:
        - period (Timestamp)
        - revenue
        - optional: cost, gross_profit, gross_margin_pct
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain 'date' column.")
    if "amount" not in df.columns:
        raise ValueError("DataFrame must contain 'amount' column.")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("'date' column must be datetime for time-series metrics.")

    df_idx = df.set_index("date").sort_index()

    grouped = df_idx["amount"].resample(freq).sum().rename("revenue")
    ts = grouped.to_frame()

    if "cost" in df_idx.columns:
        cost_series = df_idx["cost"].resample(freq).sum().rename("cost")
        ts = ts.join(cost_series, how="left")
        ts["gross_profit"] = ts["revenue"] - ts["cost"]
        ts["gross_margin_pct"] = np.where(
            ts["revenue"] > 0,
            (ts["gross_profit"] / ts["revenue"]) * 100.0,
            np.nan,
        )

    ts = ts.reset_index().rename(columns={"date": "period"})
    return ts


# ---------------------------------------------------------------------------
# Segment metrics
# ---------------------------------------------------------------------------


def compute_segment_revenue(
    df: pd.DataFrame, segment_col: str, top_n: int = 5
) -> pd.DataFrame:
    """Compute revenue and related metrics grouped by a segment column.

    Args:
        df: Cleaned transactions DataFrame.
        segment_col: Column name to group by (e.g. 'category', 'product',
            'customer_id', 'channel').
        top_n: Number of top segments to return (by revenue). If None, return all.

    Returns:
        DataFrame with columns:
        - <segment_col>
        - revenue
        - optional: cost, gross_profit, gross_margin_pct
        - order_count
        - revenue_share_pct
    """
    if segment_col not in df.columns:
        raise ValueError(
            f"DataFrame does not contain segment column '{segment_col}'."
        )

    if "amount" not in df.columns:
        raise ValueError("DataFrame must contain 'amount' column.")

    group = df.groupby(segment_col)

    revenue = group["amount"].sum().rename("revenue")
    result = revenue.to_frame()

    if "cost" in df.columns:
        cost = group["cost"].sum().rename("cost")
        result = result.join(cost, how="left")
        result["gross_profit"] = result["revenue"] - result["cost"]
        result["gross_margin_pct"] = np.where(
            result["revenue"] > 0,
            (result["gross_profit"] / result["revenue"]) * 100.0,
            np.nan,
        )

    result["order_count"] = group["amount"].count()

    total_revenue = result["revenue"].sum()
    if total_revenue > 0:
        result["revenue_share_pct"] = (result["revenue"] / total_revenue) * 100.0
    else:
        result["revenue_share_pct"] = 0.0

    result = result.sort_values("revenue", ascending=False)

    if top_n is not None:
        result = result.head(top_n)

    result = result.reset_index()  # bring segment column out of index
    return result


# ---------------------------------------------------------------------------
# Trend metrics (monthly trend)
# ---------------------------------------------------------------------------


def compute_monthly_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute simple month-over-month revenue trend.

    Uses monthly aggregated revenue and compares the last month with the
    previous month (with non-zero revenue).

    Args:
        df: Cleaned transactions DataFrame.

    Returns:
        dict with keys:
        - has_enough_data (bool)
        - current_period
        - current_revenue
        - previous_period
        - previous_revenue
        - absolute_change
        - percent_change
        - direction ("up", "down", "flat", or "insufficient_data")
    """
    ts_monthly = compute_revenue_time_series(df, freq="M")
    ts_monthly = ts_monthly.sort_values("period")

    # Consider months with non-null, non-zero revenue
    ts_valid = ts_monthly[
        ts_monthly["revenue"].notna() & (ts_monthly["revenue"] != 0)
    ]

    if len(ts_valid) < 2:
        return {
            "has_enough_data": False,
            "direction": "insufficient_data",
            "current_period": None,
            "current_revenue": None,
            "previous_period": None,
            "previous_revenue": None,
            "absolute_change": None,
            "percent_change": None,
        }

    last = ts_valid.iloc[-1]
    prev = ts_valid.iloc[-2]

    current_rev = float(last["revenue"])
    previous_rev = float(prev["revenue"])
    abs_change = current_rev - previous_rev
    pct_change: Optional[float] = None
    if previous_rev != 0:
        pct_change = (abs_change / previous_rev) * 100.0

    if abs_change > 0:
        direction = "up"
    elif abs_change < 0:
        direction = "down"
    else:
        direction = "flat"

    return {
        "has_enough_data": True,
        "direction": direction,
        "current_period": last["period"],
        "current_revenue": current_rev,
        "previous_period": prev["period"],
        "previous_revenue": previous_rev,
        "absolute_change": abs_change,
        "percent_change": pct_change,
    }


# ---------------------------------------------------------------------------
# Anomaly detection (simple daily z-score)
# ---------------------------------------------------------------------------


def detect_revenue_anomaly_last_day(
    df: pd.DataFrame,
    lookback_days: int = 30,
    sigma_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Detect if the last day's revenue is anomalous vs recent history.

    Approach:
    - Compute daily revenue via resampling.
    - Consider the last available day as "current".
    - Use the previous `lookback_days` days as history (if available).
    - Compute mean, std, and z-score of current vs history.
    - Flag as "high", "low", or "normal" if |z| exceeds sigma_threshold.

    Args:
        df: Cleaned transactions DataFrame.
        lookback_days: How many days of history to consider (max).
        sigma_threshold: Z-score threshold for flagging anomalies.

    Returns:
        dict with keys:
        - has_enough_history (bool)
        - current_date
        - current_revenue
        - history_mean
        - history_std
        - z_score
        - status ("high", "low", "normal", or "insufficient_history")
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain 'date' column.")
    if "amount" not in df.columns:
        raise ValueError("DataFrame must contain 'amount' column.")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("'date' column must be datetime for anomaly detection.")

    df_idx = df.set_index("date").sort_index()
    daily = df_idx["amount"].resample("D").sum()

    if daily.empty:
        return {
            "has_enough_history": False,
            "status": "insufficient_history",
            "current_date": None,
            "current_revenue": None,
            "history_mean": None,
            "history_std": None,
            "z_score": None,
        }

    last_date = daily.index.max()
    current_revenue = float(daily.loc[last_date])

    # History window includes up to (lookback_days) days BEFORE last_date
    window_start = last_date - pd.Timedelta(days=lookback_days)
    history_window = daily.loc[window_start:last_date]

    # Exclude the current day from history
    if last_date in history_window.index and len(history_window) > 1:
        history_window = history_window.drop(last_date)

    history = history_window.dropna()

    if len(history) < 5:
        # Require at least 5 days of history to consider it meaningful
        return {
            "has_enough_history": False,
            "status": "insufficient_history",
            "current_date": last_date,
            "current_revenue": current_revenue,
            "history_mean": None,
            "history_std": None,
            "z_score": None,
        }

    mean = float(history.mean())
    std = float(history.std(ddof=0))  # population std

    if std == 0:
        z_score = None
        status = "normal"
    else:
        z_score = (current_revenue - mean) / std
        if z_score > sigma_threshold:
            status = "high"
        elif z_score < -sigma_threshold:
            status = "low"
        else:
            status = "normal"

    return {
        "has_enough_history": True,
        "status": status,
        "current_date": last_date,
        "current_revenue": current_revenue,
        "history_mean": mean,
        "history_std": std,
        "z_score": z_score,
    }


# ---------------------------------------------------------------------------
# Metrics bundle orchestrator
# ---------------------------------------------------------------------------


def build_metrics_bundle(df: pd.DataFrame, config: Config) -> Dict[str, Any]:
    """Build a composite metrics bundle for use by the insight generator.

    The bundle includes:
    - overall: dict of overall metrics
    - time_series: DataFrame of revenue over time at configured granularity
    - segments: dict of DataFrames keyed by segment type
        - category, product, customer_id, channel (if columns present)
    - monthly_trend: dict with month-over-month trend info
    - anomaly: dict with last-day anomaly info

    Args:
        df: Cleaned transactions DataFrame.
        config: Loaded configuration.

    Returns:
        dict representing all key metrics.
    """
    overall = compute_overall_metrics(df)

    freq = _freq_from_granularity(config.analytics.time_granularity)
    time_series = compute_revenue_time_series(df, freq=freq)

    segments: Dict[str, Any] = {}
    top_n = config.analytics.top_n

    for seg in ["category", "product", "customer_id", "channel"]:
        if seg in df.columns:
            segments[seg] = compute_segment_revenue(df, seg, top_n=top_n)

    monthly_trend = compute_monthly_trend(df)

    anomaly = detect_revenue_anomaly_last_day(
        df,
        lookback_days=config.analytics.anomaly_lookback_days,
        sigma_threshold=config.analytics.anomaly_sigma_threshold,
    )

    return {
        "overall": overall,
        "time_series": time_series,
        "segments": segments,
        "monthly_trend": monthly_trend,
        "anomaly": anomaly,
    }
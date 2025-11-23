"""Demo script for running analytics on the sample dataset."""

from fig.preprocessing import load_and_clean_transactions
from fig.analytics import (
    compute_overall_metrics,
    compute_revenue_time_series,
    compute_segment_revenue,
    compute_monthly_trend,
    detect_revenue_anomaly_last_day,
    build_metrics_bundle,
)


def main() -> None:
    df, report, cfg = load_and_clean_transactions("config.yaml")

    print("\n=== Data Pipeline Reports ===")
    print("Validation:")
    for k, v in report["validation"].items():
        print(f"  {k}: {v}")
    print("Cleaning:")
    for k, v in report["cleaning"].items():
        print(f"  {k}: {v}")

    # --- Overall metrics ---
    overall = compute_overall_metrics(df)
    print("\n=== Overall Metrics ===")
    print(f"Date range       : {overall['date_min'].date()} -> {overall['date_max'].date()}")
    print(f"Total revenue    : {overall['total_revenue']:.2f}")
    if overall["total_cost"] is not None:
        print(f"Total cost       : {overall['total_cost']:.2f}")
    if overall["gross_profit"] is not None:
        print(f"Gross profit     : {overall['gross_profit']:.2f}")
    if overall["gross_margin_pct"] is not None:
        print(f"Gross margin %   : {overall['gross_margin_pct']:.2f}%")
    print(f"Transactions     : {overall['n_transactions']}")
    if overall["avg_order_value"] is not None:
        print(f"Avg order value  : {overall['avg_order_value']:.2f}")

    # --- Time series (at configured granularity) ---
    print("\n=== Time Series (Configured Granularity) ===")
    freq = cfg.analytics.time_granularity
    if freq.lower() == "day":
        freq_label = "Daily"
    elif freq.lower() == "week":
        freq_label = "Weekly"
    else:
        freq_label = "Monthly"

    ts = compute_revenue_time_series(
        df,
        freq={
            "day": "D",
            "week": "W",
            "month": "M",
        }[cfg.analytics.time_granularity.lower()],
    )
    print(f"{freq_label} revenue (first 5 rows):")
    print(ts.head())

    # --- Segment metrics: category ---
    if "category" in df.columns:
        print("\n=== Top Categories ===")
        top_categories = compute_segment_revenue(
            df, "category", top_n=cfg.analytics.top_n
        )
        print(top_categories)

    # --- Monthly trend ---
    trend = compute_monthly_trend(df)
    print("\n=== Monthly Trend ===")
    if not trend["has_enough_data"]:
        print("Not enough data to compute monthly trend.")
    else:
        direction_symbol = {
            "up": "↑",
            "down": "↓",
            "flat": "→",
        }.get(trend["direction"], "?")
        print(
            f"Last month ({trend['previous_period'].date()} -> {trend['current_period'].date()}): "
            f"{direction_symbol}"
        )
        print(f"  Previous revenue: {trend['previous_revenue']:.2f}")
        print(f"  Current revenue : {trend['current_revenue']:.2f}")
        if trend["percent_change"] is not None:
            print(f"  Change          : {trend['absolute_change']:.2f} "
                  f"({trend['percent_change']:.2f}%)")

    # --- Anomaly detection ---
    anomaly = detect_revenue_anomaly_last_day(
        df,
        lookback_days=cfg.analytics.anomaly_lookback_days,
        sigma_threshold=cfg.analytics.anomaly_sigma_threshold,
    )
    print("\n=== Last Day Revenue Anomaly Check ===")
    if not anomaly["has_enough_history"]:
        print("Not enough history for anomaly detection.")
    else:
        print(f"Last day        : {anomaly['current_date'].date()}")
        print(f"Revenue         : {anomaly['current_revenue']:.2f}")
        print(f"History mean    : {anomaly['history_mean']:.2f}")
        print(f"History std     : {anomaly['history_std']:.2f}")
        print(f"Z-score         : {anomaly['z_score']:.2f}")
        print(f"Status          : {anomaly['status']}")

    # --- Metrics bundle (for future insights use) ---
    metrics_bundle = build_metrics_bundle(df, cfg)
    print("\n=== Metrics Bundle Keys (for insights) ===")
    print("Top-level keys:", list(metrics_bundle.keys()))
    print("Segment keys   :", list(metrics_bundle["segments"].keys()))


if __name__ == "__main__":
    main()
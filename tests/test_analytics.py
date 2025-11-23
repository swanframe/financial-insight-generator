import pytest

from fig.analytics import compute_overall_metrics, compute_segment_revenue


def test_compute_overall_metrics_matches_df(clean_df):
    metrics = compute_overall_metrics(clean_df)

    assert metrics["n_transactions"] == len(clean_df)
    assert metrics["total_revenue"] > 0
    assert metrics["date_min"] <= metrics["date_max"]

    # avg_order_value should be consistent with total / count
    expected_aov = metrics["total_revenue"] / metrics["n_transactions"]
    assert pytest.approx(metrics["avg_order_value"], rel=1e-6) == expected_aov


def test_compute_segment_revenue_category(clean_df, config_obj):
    if "category" not in clean_df.columns:
        pytest.skip("No category column in sample data; skipping segment test.")

    seg_df = compute_segment_revenue(
        clean_df, "category", top_n=config_obj.analytics.top_n
    )

    assert not seg_df.empty
    assert "category" in seg_df.columns
    assert "revenue" in seg_df.columns
    assert "order_count" in seg_df.columns
def test_load_config_basic(config_obj):
    # basic sanity checks
    assert config_obj.data.input_path.name == "sample_transactions.csv"
    assert config_obj.columns.date == "order_date"
    assert config_obj.columns.amount == "total_price"

    # analytics defaults
    assert config_obj.analytics.top_n > 0
    assert config_obj.analytics.time_granularity in {"day", "week", "month"}
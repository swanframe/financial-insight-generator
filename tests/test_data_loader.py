from fig.data_loader import load_transactions


def test_load_transactions_with_sample(config_obj):
    df = load_transactions(config_obj.data.input_path)
    assert not df.empty
    # raw column names from the sample CSV
    assert "order_date" in df.columns
    assert "total_price" in df.columns
import copy

import pandas as pd
import pytest

from fig.retrieval import (
    RetrievalStatus,
    build_transaction_index_from_dataframe,
    is_retrieval_enabled,
    retrieve_transactions_for_query,
)
from fig.vector_store import TransactionMatch, VectorStoreConfigError, VectorStoreError


def test_is_retrieval_enabled_respects_flag(config_obj):
    cfg = copy.deepcopy(config_obj)
    # The helper should reflect the underlying flag.
    cfg.vector_store.enabled = False
    assert is_retrieval_enabled(cfg) is False
    cfg.vector_store.enabled = True
    assert is_retrieval_enabled(cfg) is True


def test_build_transaction_index_skips_when_disabled(config_obj):
    cfg = copy.deepcopy(config_obj)
    cfg.vector_store.enabled = False

    df = pd.DataFrame({"date": ["2024-01-01"], "amount": [100.0]})

    status = build_transaction_index_from_dataframe(
        df,
        config=cfg,
        rebuild=False,
        fail_silently=True,
    )

    assert isinstance(status, RetrievalStatus)
    assert status.attempted is False
    assert status.success is False
    assert "disabled" in (status.message or "").lower()


def test_build_transaction_index_calls_vector_store(monkeypatch, config_obj):
    cfg = copy.deepcopy(config_obj)
    cfg.vector_store.enabled = True

    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "amount": [100.0, 200.0],
        }
    )

    called = {}

    def fake_build_or_update_index(df, config, rebuild):
        called["df_shape"] = df.shape
        called["config"] = config
        called["rebuild"] = rebuild

    monkeypatch.setattr(
        "fig.retrieval.build_or_update_index",
        fake_build_or_update_index,
    )

    status = build_transaction_index_from_dataframe(
        df,
        config=cfg,
        rebuild=True,
        fail_silently=False,
    )

    assert status.attempted is True
    assert status.success is True
    assert status.message is None
    assert called["df_shape"][0] == 2
    assert called["config"] is cfg
    assert called["rebuild"] is True


def test_build_transaction_index_handles_error_when_silent(monkeypatch, config_obj):
    cfg = copy.deepcopy(config_obj)
    cfg.vector_store.enabled = True

    df = pd.DataFrame({"date": ["2024-01-01"], "amount": [100.0]})

    def fake_build_or_update_index(df, config, rebuild):
        raise VectorStoreError("simulated failure")

    monkeypatch.setattr(
        "fig.retrieval.build_or_update_index",
        fake_build_or_update_index,
    )

    status = build_transaction_index_from_dataframe(
        df,
        config=cfg,
        rebuild=False,
        fail_silently=True,
    )

    assert status.attempted is True
    assert status.success is False
    assert "simulated failure" in (status.message or "")


def test_build_transaction_index_raises_error_when_not_silent(monkeypatch, config_obj):
    cfg = copy.deepcopy(config_obj)
    cfg.vector_store.enabled = True

    df = pd.DataFrame({"date": ["2024-01-01"], "amount": [100.0]})

    def fake_build_or_update_index(df, config, rebuild):
        raise VectorStoreError("simulated failure")

    monkeypatch.setattr(
        "fig.retrieval.build_or_update_index",
        fake_build_or_update_index,
    )

    with pytest.raises(VectorStoreError):
        build_transaction_index_from_dataframe(
            df,
            config=cfg,
            rebuild=False,
            fail_silently=False,
        )


def test_retrieve_transactions_returns_none_when_disabled(config_obj):
    cfg = copy.deepcopy(config_obj)
    cfg.vector_store.enabled = False

    ctx = retrieve_transactions_for_query(
        "anything",
        config=cfg,
        top_k=None,
        language="en",
        fail_silently=True,
    )

    assert ctx is None


def test_retrieve_transactions_raises_when_disabled_and_not_silent(config_obj):
    cfg = copy.deepcopy(config_obj)
    cfg.vector_store.enabled = False

    with pytest.raises(VectorStoreConfigError):
        retrieve_transactions_for_query(
            "anything",
            config=cfg,
            top_k=None,
            language="en",
            fail_silently=False,
        )


def test_retrieve_transactions_wraps_matches_into_context(monkeypatch, config_obj):
    cfg = copy.deepcopy(config_obj)
    cfg.vector_store.enabled = True

    fake_matches = [
        TransactionMatch(
            id="T1",
            score=0.5,
            metadata={"transaction_id": "T1", "date": "2024-03-01", "amount": 123.0},
            text="Example doc",
        )
    ]

    def fake_query_similar_transactions(query_text, config, top_k):
        # Basic sanity checks on arguments passed through.
        assert query_text == "large transactions"
        assert config is cfg
        assert top_k == 3
        return fake_matches

    monkeypatch.setattr(
        "fig.retrieval.query_similar_transactions",
        fake_query_similar_transactions,
    )

    ctx = retrieve_transactions_for_query(
        "large transactions",
        config=cfg,
        top_k=3,
        language="en",
        filters={"month": "2024-03"},
        fail_silently=False,
    )

    assert ctx is not None
    assert ctx.query == "large transactions"
    assert ctx.language == "en"
    assert ctx.filters["month"] == "2024-03"
    assert len(ctx.matches) == 1
    m = ctx.matches[0]
    assert m.transaction_id == "T1"
    assert m.date == "2024-03-01"
    assert m.amount == 123.0
    assert m.text == "Example doc"


def test_retrieve_transactions_returns_none_on_error_when_silent(monkeypatch, config_obj):
    cfg = copy.deepcopy(config_obj)
    cfg.vector_store.enabled = True

    def fake_query_similar_transactions(query_text, config, top_k):
        raise VectorStoreError("simulated query error")

    monkeypatch.setattr(
        "fig.retrieval.query_similar_transactions",
        fake_query_similar_transactions,
    )

    ctx = retrieve_transactions_for_query(
        "anything",
        config=cfg,
        top_k=None,
        language="en",
        fail_silently=True,
    )

    assert ctx is None


def test_retrieve_transactions_raises_on_error_when_not_silent(monkeypatch, config_obj):
    cfg = copy.deepcopy(config_obj)
    cfg.vector_store.enabled = True

    def fake_query_similar_transactions(query_text, config, top_k):
        raise VectorStoreError("simulated query error")

    monkeypatch.setattr(
        "fig.retrieval.query_similar_transactions",
        fake_query_similar_transactions,
    )

    with pytest.raises(VectorStoreError):
        retrieve_transactions_for_query(
            "anything",
            config=cfg,
            top_k=None,
            language="en",
            fail_silently=False,
        )
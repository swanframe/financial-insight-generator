import pandas as pd
import pytest

from fig.config import load_config, Config
from fig import vector_store


def _make_base_config(tmp_path, extra_vector_yaml: str = "") -> Config:
    """Create a minimal config file for vector store tests."""
    cfg_path = tmp_path / "config_vector_test.yaml"
    cfg_text = f"""
data:
  input_path: "data/raw/sample.csv"
columns:
  date: "date"
  amount: "amount"
  category: "category"
  product: "product"
vector_store:
  enabled: true
  provider: "in_memory"
{extra_vector_yaml}
"""
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return load_config(cfg_path)


def test_build_index_raises_when_disabled(tmp_path):
    """If vector_store.enabled is false, building the index should raise a config error."""
    cfg_path = tmp_path / "config_vector_disabled.yaml"
    cfg_path.write_text(
        """
data:
  input_path: "data/raw/sample.csv"
columns:
  date: "date"
  amount: "amount"
vector_store:
  enabled: false
  provider: "in_memory"
        """,
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)

    df = pd.DataFrame(
        [
            {"date": "2024-01-01", "amount": 100.0},
        ]
    )

    with pytest.raises(vector_store.VectorStoreConfigError):
        vector_store.build_or_update_index(df, cfg)


def test_build_and_query_in_memory_store(monkeypatch, tmp_path):
    """Build an in-memory index and query similar transactions using a fake embedding function."""
    cfg = _make_base_config(tmp_path)

    # Simple example DataFrame with internal column names already applied.
    df = pd.DataFrame(
        [
            {
                "date": "2024-03-10",
                "amount": 1200000.0,
                "category": "Electronics",
                "product": "Wireless Headphones",
            },
            {
                "date": "2024-03-11",
                "amount": 80000.0,
                "category": "Groceries",
                "product": "Fresh Apples",
            },
        ]
    )

    # Ensure we start from a clean in-memory store for this test.
    vector_store._IN_MEMORY_STORES.clear()

    # Fake embedding function:
    # - Documents mentioning "headphones" map to [1, 0]
    # - Documents mentioning "apples" map to [0, 1]
    # - Others map to [0, 0]
    # - Query "headphones" should map to [1, 0], making the first row the top match.
    def fake_embed_texts(texts, config):
        vectors = []
        for t in texts:
            t_lower = t.lower()
            if "headphones" in t_lower:
                vectors.append([1.0, 0.0])
            elif "apples" in t_lower:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.0, 0.0])
        return vectors

    # Monkeypatch the embed_texts helper inside the vector_store module.
    monkeypatch.setattr(
        "fig.vector_store.embed_texts",
        fake_embed_texts,
        raising=True,
    )

    # Build the index.
    vector_store.build_or_update_index(df, cfg, rebuild=True)

    # Query for "headphones" and expect the electronics transaction to come first.
    matches = vector_store.query_similar_transactions(
        "customer bought new headphones",
        config=cfg,
        top_k=1,
    )

    assert len(matches) == 1
    match = matches[0]

    assert isinstance(match.score, float)
    assert match.metadata["product"] == "Wireless Headphones"
    assert match.metadata["category"] == "Electronics"
    assert match.metadata["amount"] == pytest.approx(1200000.0)


def test_query_in_memory_store_empty_index(monkeypatch, tmp_path):
    """Querying an empty in-memory store should simply return an empty list."""
    cfg = _make_base_config(tmp_path)

    # Clear any existing in-memory state.
    vector_store._IN_MEMORY_STORES.clear()

    def fake_embed_texts(texts, config):
        # Deterministic embedding; content doesn't matter for this test.
        return [[0.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        "fig.vector_store.embed_texts",
        fake_embed_texts,
        raising=True,
    )

    matches = vector_store.query_similar_transactions(
        "any query text",
        config=cfg,
        top_k=5,
    )
    assert matches == []
from pathlib import Path

import pytest

import run_vector_search
from fig.vector_store import TransactionMatch


def _write_config(tmp_path: Path, extra_vector_yaml: str) -> Path:
    """Write a minimal config file with a vector_store section."""
    cfg_path = tmp_path / "config_vector_search.yaml"
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
    return cfg_path


def test_search_cli_handles_disabled_vector_store(tmp_path, capsys):
    """When vector_store.enabled = false, the CLI should print a friendly note and exit."""
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

    run_vector_search.main(
        [
            "--config",
            str(cfg_path),
            "--query",
            "any query",
        ]
    )

    captured = capsys.readouterr()
    out = captured.out
    # We rely on the default English fallback string here.
    assert "Vector store is disabled" in out


def test_search_cli_calls_vector_store_and_prints_results(monkeypatch, tmp_path, capsys):
    """The CLI should call vector_store.query_similar_transactions and print formatted results."""
    cfg_path = _write_config(tmp_path, extra_vector_yaml="")

    captured = {}

    def fake_query_similar(query_text, config, top_k=None):
        captured["query_text"] = query_text
        captured["top_k"] = top_k

        return [
            TransactionMatch(
                id="txn-0",
                score=0.95,
                metadata={
                    "transaction_id": "txn-0",
                    "date": "2024-03-10",
                    "amount": 1200000.0,
                    "category": "Electronics",
                    "product": "Wireless Headphones",
                },
                text="Transaction txn-0 on 2024-03-10 in category Electronics for product Wireless Headphones.",
            ),
            TransactionMatch(
                id="txn-1",
                score=0.80,
                metadata={
                    "transaction_id": "txn-1",
                    "date": "2024-03-11",
                    "amount": 80000.0,
                    "category": "Groceries",
                    "product": "Fresh Apples",
                },
                text="Transaction txn-1 on 2024-03-11 in category Groceries for product Fresh Apples.",
            ),
        ]

    # Patch the vector_store query used inside the CLI to avoid any real index calls.
    monkeypatch.setattr(
        "run_vector_search.vector_store.query_similar_transactions",
        fake_query_similar,
        raising=True,
    )

    run_vector_search.main(
        [
            "--config",
            str(cfg_path),
            "--query",
            "customer bought new headphones",
            "--top-k",
            "3",
        ]
    )

    captured_io = capsys.readouterr()
    out = captured_io.out

    # Ensure our fake query function was invoked correctly.
    assert captured["query_text"] == "customer bought new headphones"
    assert captured["top_k"] == 3

    # And ensure the CLI printed some of the expected fields from the metadata.
    assert "Wireless Headphones" in out
    assert "Electronics" in out
    assert "1200000.0" in out
    assert "similarity=" in out
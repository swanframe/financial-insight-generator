import math

from fig.vector_store import TransactionMatch
from fig.retrieval_schema import (
    RetrievedTransaction,
    RetrievalContext,
    build_retrieval_context,
)


def test_build_retrieval_context_populates_fields():
    matches = [
        TransactionMatch(
            id="T1",
            score=0.9,
            metadata={
                "transaction_id": "T1",
                "date": "2024-03-01",
                "amount": 123.45,
                "category": "Electronics",
                "product": "Headphones",
                "customer_id": "C1",
                "channel": "Online",
                "row_index": 5,
            },
            text="Example transaction",
        )
    ]

    ctx = build_retrieval_context(
        matches,
        query="large transactions in March",
        language="en",
        top_k=1,
        filters={"month": "2024-03"},
    )

    assert isinstance(ctx, RetrievalContext)
    assert ctx.query == "large transactions in March"
    assert ctx.language == "en"
    assert ctx.effective_top_k() == 1
    assert ctx.filters["month"] == "2024-03"
    assert len(ctx.matches) == 1

    m = ctx.matches[0]
    assert isinstance(m, RetrievedTransaction)
    assert m.id == "T1"
    assert m.transaction_id == "T1"
    assert m.date == "2024-03-01"
    assert math.isclose(m.amount, 123.45)
    assert m.category == "Electronics"
    assert m.product == "Headphones"
    assert m.customer_id == "C1"
    assert m.channel == "Online"
    assert m.row_index == 5
    assert m.text == "Example transaction"

    prompt_dict = ctx.to_prompt_dict()
    assert prompt_dict["query"] == "large transactions in March"
    assert prompt_dict["top_k"] == 1
    assert len(prompt_dict["matches"]) == 1
    prompt_match = prompt_dict["matches"][0]
    assert prompt_match["transaction_id"] == "T1"
    assert prompt_match["date"] == "2024-03-01"
    assert prompt_match["amount"] == 123.45
    assert prompt_match["category"] == "Electronics"
    assert prompt_match["product"] == "Headphones"


def test_build_retrieval_context_handles_missing_metadata():
    matches = [
        TransactionMatch(
            id="T2",
            score=0.5,
            metadata={},
            text=None,
        )
    ]

    ctx = build_retrieval_context(matches, query=None, language="en")

    assert ctx.effective_top_k() == 1
    assert len(ctx.matches) == 1

    m = ctx.matches[0]
    # When metadata is missing, transaction_id should fall back to id.
    assert m.transaction_id == "T2"
    assert m.date is None
    assert m.amount is None
    assert m.category is None
    assert m.product is None
    assert m.customer_id is None
    assert m.channel is None
    assert m.metadata == {}
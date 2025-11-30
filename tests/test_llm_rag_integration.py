import copy

from fig.llm_insights import generate_llm_report
from fig.llm_chatbot import answer_freeform_question
from fig.retrieval_schema import RetrievedTransaction, RetrievalContext


def _fake_retrieval_context(language: str) -> RetrievalContext:
    match = RetrievedTransaction(
        id="T1",
        score=0.9,
        transaction_id="T1",
        date="2024-03-01",
        amount=123.45,
        category="Electronics",
        product="Headphones",
        customer_id="C1",
        channel="Online",
        row_index=5,
        text="Example transaction",
        metadata={},
    )
    return RetrievalContext(
        query="test query",
        matches=[match],
        top_k=1,
        language=language,
        filters={"month": "2024-03"},
    )


def test_llm_report_prompt_includes_rag_context(monkeypatch, metrics_bundle, config_obj):
    cfg = copy.deepcopy(config_obj)
    cfg.llm.enabled = True
    cfg.llm.mode = "llm"
    cfg.vector_store.enabled = True

    captured = {}

    def fake_generate_text(user_prompt, *args, **kwargs):
        captured["user_prompt"] = user_prompt
        return "LLM report text"

    # Patch the LLM client used inside llm_insights.
    monkeypatch.setattr("fig.llm_insights.llm_client.generate_text", fake_generate_text)

    # Patch retrieval so we don't depend on the real vector store in this test.
    def fake_retrieve_transactions_for_query(*args, **kwargs):
        return _fake_retrieval_context(language="en")

    monkeypatch.setattr(
        "fig.llm_insights.retrieve_transactions_for_query",
        fake_retrieve_transactions_for_query,
    )

    report = generate_llm_report(metrics_bundle, cfg, language="en")

    assert report == "LLM report text"
    assert "[RAG context: sample transactions]" in captured["user_prompt"]
    assert "similarity=" in captured["user_prompt"]
    # The question-less RAG helper should still mention dates/categories/products.
    assert "category=" in captured["user_prompt"]
    assert "product=" in captured["user_prompt"]


def test_chat_prompt_includes_rag_context(monkeypatch, metrics_bundle, config_obj):
    cfg = copy.deepcopy(config_obj)
    cfg.llm.enabled = True
    cfg.vector_store.enabled = True

    captured = {}

    def fake_generate_text(user_prompt, *args, **kwargs):
        captured["user_prompt"] = user_prompt
        return "LLM answer"

    monkeypatch.setattr("fig.llm_chatbot.llm_client.generate_text", fake_generate_text)

    def fake_retrieve_transactions_for_query(*args, **kwargs):
        return _fake_retrieval_context(language="en")

    monkeypatch.setattr(
        "fig.llm_chatbot.retrieve_transactions_for_query",
        fake_retrieve_transactions_for_query,
    )

    context = {
        "metrics": metrics_bundle,
        "config": cfg,
        "language": "en",
    }

    answer = answer_freeform_question("How is revenue trending?", context)

    assert answer == "LLM answer"
    assert "[RAG context: relevant transactions]" in captured["user_prompt"]
    assert "similarity=" in captured["user_prompt"]
    # Ensure the original question is still present in the prompt as well.
    assert "How is revenue trending?" in captured["user_prompt"]
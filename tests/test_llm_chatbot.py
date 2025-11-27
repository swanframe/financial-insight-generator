import copy

from fig import llm_chatbot
from fig.llm_client import LlmConfigError


def test_answer_freeform_question_uses_llm_when_enabled(
    monkeypatch,
    metrics_bundle,
    config_obj,
):
    """When LLM is enabled, the helper should call generate_text and return its answer."""
    cfg = copy.deepcopy(config_obj)
    cfg.llm.enabled = True  # turn on LLM features for this test

    captured = {}

    def fake_generate_text(user_prompt, *, config, system_prompt=None, language=None):
        captured["user_prompt"] = user_prompt
        captured["system_prompt"] = system_prompt
        captured["language"] = language
        return "Fake LLM answer"

    # Patch the global generate_text used by llm_chatbot
    monkeypatch.setattr("fig.llm_client.generate_text", fake_generate_text)

    context = {
        "df": None,  # not used by llm_chatbot
        "metrics": metrics_bundle,
        "config": cfg,
        "language": "en",
    }

    question = "How is revenue trending?"
    answer = llm_chatbot.answer_freeform_question(question, context)

    assert answer == "Fake LLM answer"
    # Basic sanity checks on the prompt contents
    assert "Structured metrics summary:" in captured["user_prompt"]
    assert question in captured["user_prompt"]
    assert captured["language"] == "en"
    assert "Answer the user's question" in captured["user_prompt"] or "Answer the user's" in captured["user_prompt"]


def test_answer_freeform_question_handles_llm_config_error(
    monkeypatch,
    metrics_bundle,
    config_obj,
):
    """Configuration errors from llm_client should be turned into a friendly note."""
    cfg = copy.deepcopy(config_obj)
    cfg.llm.enabled = True  # pretend LLM is enabled, but misconfigured

    def fake_generate_text(*args, **kwargs):
        raise LlmConfigError("simulated missing API key")

    monkeypatch.setattr("fig.llm_client.generate_text", fake_generate_text)

    context = {
        "df": None,
        "metrics": metrics_bundle,
        "config": cfg,
        "language": "en",
    }

    answer = llm_chatbot.answer_freeform_question("Any question", context)
    assert isinstance(answer, str)
    assert "missing API key" in answer or "simulated missing API key" in answer
    assert "LLM" in answer or "Note" in answer
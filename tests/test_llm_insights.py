import copy

from fig import insights, llm_client
from fig.llm_insights import generate_llm_report


def test_generate_llm_report_falls_back_when_llm_disabled(metrics_bundle, config_obj):
    """When LLM is disabled, the function should still return a report string and not error."""
    cfg = copy.deepcopy(config_obj)
    # Ensure we are in a mode that *would* try to use the LLM if it were enabled.
    cfg.llm.mode = "llm"
    cfg.llm.enabled = False

    report = generate_llm_report(metrics_bundle, cfg, language="en")
    assert isinstance(report, str)
    # We expect a short note indicating that LLM features are disabled or unavailable.
    assert "[Note]" in report or "[Catatan]" in report


def test_generate_llm_report_template_mode_skips_llm(monkeypatch, metrics_bundle, config_obj):
    """In template mode, generate_llm_report should behave like generate_full_report and not call the LLM."""
    cfg = copy.deepcopy(config_obj)
    cfg.llm.enabled = True
    cfg.llm.mode = "llm"  # but we will override to 'template' at the call site

    called = {"value": False}

    def fake_generate_text(*args, **kwargs):
        called["value"] = True
        return "SHOULD NOT BE USED"

    # Patch the generate_text used inside llm_insights
    monkeypatch.setattr("fig.llm_insights.llm_client.generate_text", fake_generate_text)

    template_report = insights.generate_full_report(metrics_bundle, language="en")
    report = generate_llm_report(
        metrics_bundle,
        cfg,
        language="en",
        mode_override="template",
    )

    # Should match the template path exactly and never call the LLM.
    assert report == template_report
    assert called["value"] is False


def test_generate_llm_report_hybrid_falls_back_on_error(monkeypatch, metrics_bundle, config_obj):
    """In hybrid mode, LLM errors should result in a note + template-based fallback."""
    cfg = copy.deepcopy(config_obj)
    cfg.llm.enabled = True
    cfg.llm.mode = "hybrid"

    def fake_generate_text(*args, **kwargs):
        raise llm_client.LlmError("simulated provider failure")

    monkeypatch.setattr("fig.llm_insights.llm_client.generate_text", fake_generate_text)

    template_report = insights.generate_full_report(metrics_bundle, language="en")
    report = generate_llm_report(metrics_bundle, cfg, language="en")

    assert isinstance(report, str)
    # We should see a note about errors and the template report included.
    assert "[Note]" in report or "[Catatan]" in report
    assert template_report in report
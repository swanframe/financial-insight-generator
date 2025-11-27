import pandas as pd

from fig.llm_prompts import (
    MetricsSummaryOptions,
    summarize_metrics_bundle,
    build_system_prompt_for_report,
    build_user_prompt_for_report,
    build_user_prompt_for_question,
)


def test_summarize_metrics_bundle_contains_key_numbers(metrics_bundle):
    """Summarized context should contain high-level metrics such as total revenue."""
    summary = summarize_metrics_bundle(metrics_bundle)
    # Smoke-check that the summary is non-empty and contains at least one key label.
    assert isinstance(summary, str)
    assert "Overall metrics" in summary or "Total revenue" in summary
    assert "[Context truncated" not in summary


def test_summarize_metrics_bundle_respects_max_chars():
    """Summarization should truncate when max_context_chars is very small."""
    # Build a synthetic metrics bundle that is definitely long enough to be truncated.
    periods = pd.date_range("2023-01-01", periods=20, freq="D")
    ts = pd.DataFrame({"period": periods, "revenue": range(20)})
    fake_bundle = {
        "overall": {"total_revenue": 1234567890.0, "n_transactions": 9999},
        "time_series": ts,
        "segments": {},
        "monthly_trend": {},
        "anomaly": {},
    }

    opts = MetricsSummaryOptions(max_context_chars=100, max_time_series_rows=10, max_segments_per_type=5)
    summary = summarize_metrics_bundle(fake_bundle, options=opts)
    assert isinstance(summary, str)
    # We expect a truncation marker when the context is forcibly short.
    assert "[Context truncated for length]" in summary
    assert len(summary) <= 150  # some slack for the marker itself


def test_build_system_prompt_for_report_languages():
    """System prompts should clearly encode language requirements."""
    en_prompt = build_system_prompt_for_report(language="en")
    id_prompt = build_system_prompt_for_report(language="id")

    assert "English" in en_prompt
    assert "Bahasa Indonesia" in id_prompt


def test_build_user_prompt_for_report_contains_metrics_and_language(metrics_bundle):
    """User prompt for report should embed metrics summary and language hint."""
    prompt_en = build_user_prompt_for_report(metrics_bundle, language="en")
    prompt_id = build_user_prompt_for_report(metrics_bundle, language="id")

    # Both prompts should reference the structured metrics summary.
    assert "Structured metrics summary:" in prompt_en
    assert "Structured metrics summary:" in prompt_id

    # Language hints
    assert "English" in prompt_en
    assert "Bahasa Indonesia" in prompt_id


def test_build_user_prompt_for_question_includes_question_and_commands(metrics_bundle):
    """Question prompt should include the question text and optional command hints."""
    question = "Why did revenue increase last month?"
    available_commands = ["summary", "trend", "top categories"]
    prompt = build_user_prompt_for_question(
        question,
        metrics_bundle,
        language="en",
        available_commands=available_commands,
    )

    assert "User question" in prompt
    assert question in prompt
    # Commands are mentioned somewhere
    for cmd in available_commands:
        assert cmd in prompt
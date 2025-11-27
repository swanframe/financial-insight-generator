"""LLM-powered helper for the interactive chat assistant.

This module is intentionally small and focused:

- It takes a user question + the existing chat context (metrics, config, language).
- It builds an LLM prompt using fig.llm_prompts.
- It calls the provider-agnostic fig.llm_client.generate_text.
- It returns a plain string answer, handling errors gracefully.

The CLI chat loop in `chatbot.py` decides *when* to call this helper. By
default, it is only used for free-form questions that do not match any known
rule-based command (summary, trend, top categories, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from . import llm_client
from .llm_prompts import (
    MetricsSummaryOptions,
    build_system_prompt_for_chat,
    build_user_prompt_for_question,
)


def _normalise_language(language: str | None) -> str:
    """Normalise language codes to the small set we explicitly handle."""
    if not language:
        return "en"
    lang = str(language).strip().lower()
    if lang.startswith("id"):
        return "id"
    if lang.startswith("en"):
        return "en"
    return "en"


def _resolve_language(context: Dict[str, Any], cfg: Any) -> str:
    """Resolve the effective language from context or config."""
    raw_lang = context.get("language")
    if raw_lang is None and hasattr(cfg, "ui"):
        raw_lang = getattr(cfg.ui, "language", "en")
    language = str(raw_lang).strip().lower() if raw_lang else "en"
    return _normalise_language(language)


def _build_available_commands() -> Sequence[str]:
    """List CLI-style commands the model may refer to in explanations."""
    return [
        "summary",
        "overview",
        "top categories",
        "top products",
        "top customers",
        "top channels",
        "trend",
        "anomaly",
        "time series",
    ]


def _note_llm_config_error(language: str, details: str) -> str:
    """User-facing message when LLM is disabled/misconfigured."""
    lang = _normalise_language(language)
    if lang == "id":
        return (
            "[Catatan] Pertanyaan ini tidak dapat dijawab dengan LLM karena "
            "konfigurasi atau API key bermasalah. Anda tetap bisa memakai "
            "perintah bawaan seperti 'help', 'summary', atau 'top categories'.\n"
            f"Detail teknis: {details}"
        )
    return (
        "[Note] This question could not be answered using the LLM because of a "
        "configuration problem or missing API key. You can still use the "
        "built-in commands such as 'help', 'summary', or 'top categories'.\n"
        f"Technical details: {details}"
    )


def _note_llm_runtime_error(language: str, details: str) -> str:
    """User-facing message when the provider call itself fails."""
    lang = _normalise_language(language)
    if lang == "id":
        return (
            "[Catatan] Terjadi kesalahan saat memanggil penyedia LLM ketika "
            "menjawab pertanyaan ini. Coba lagi nanti, atau gunakan perintah "
            "bawaan seperti 'help', 'summary', atau 'trend'.\n"
            f"Detail teknis: {details}"
        )
    return (
        "[Note] An error occurred while calling the LLM provider when trying "
        "to answer this question. Try again later, or use built-in commands "
        "such as 'help', 'summary', or 'trend'.\n"
        f"Technical details: {details}"
    )


def answer_freeform_question(question: str, context: Dict[str, Any]) -> str:
    """Answer a free-form user question using the LLM and metrics bundle.

    Parameters
    ----------
    question:
        The raw user input that did not match any known rule-based command.
    context:
        Dictionary containing at least:
        - "metrics": metrics bundle from analytics.build_metrics_bundle
        - "config": Config object
        - "language": optional language code (e.g. "en", "id")
        Other keys (such as "df") may be present but are not required here.

    Returns
    -------
    str
        A textual answer. On configuration or provider errors, this will be a
        friendly note explaining the situation and suggesting alternatives.
    """
    metrics = context.get("metrics")
    cfg = context.get("config")

    if metrics is None or cfg is None:
        raise ValueError("LLM chat context must include 'metrics' and 'config'.")

    language = _resolve_language(context, cfg)
    llm_cfg = getattr(cfg, "llm", None)

    # If there is no LLM config or it is disabled, treat this as a configuration error.
    if llm_cfg is None or not getattr(llm_cfg, "enabled", False):
        return _note_llm_config_error(language, "LLM features are disabled (llm.enabled = false).")

    opts = MetricsSummaryOptions(max_context_chars=getattr(llm_cfg, "max_context_chars", 12_000))
    system_prompt = build_system_prompt_for_chat(language=language)
    available_commands = _build_available_commands()
    user_prompt = build_user_prompt_for_question(
        question=question,
        metrics_bundle=metrics,
        language=language,
        available_commands=available_commands,
        options=opts,
    )

    try:
        text = llm_client.generate_text(
            user_prompt,
            config=cfg,
            system_prompt=system_prompt,
            language=language,
        )
        return text.strip()
    except llm_client.LlmConfigError as exc:
        # Missing API key, unsupported provider, etc.
        return _note_llm_config_error(language, str(exc))
    except llm_client.LlmError as exc:
        # Network issues, provider-side failures, etc.
        return _note_llm_runtime_error(language, str(exc))
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return _note_llm_runtime_error(language, str(exc))
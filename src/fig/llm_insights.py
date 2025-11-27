"""LLM-powered insight generation for Financial Insight Generator.

This module provides a higher-level report generator that mirrors the existing
template-based ``generate_full_report`` function, but uses an LLM underneath
when enabled.

The main entry point is :func:`generate_llm_report`, which:

- Accepts the same ``metrics_bundle`` used by the template engine.
- Reads behaviour flags from ``config.llm`` (enabled, mode, etc.).
- Falls back gracefully to the template-based report if LLM usage is disabled
  or fails for any reason.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .config import Config
from . import insights
from . import llm_client
from .llm_prompts import (
    MetricsSummaryOptions,
    build_system_prompt_for_report,
    build_user_prompt_for_report,
)


def _is_indonesian(language: str | None) -> bool:
    if not language:
        return False
    return str(language).strip().lower().startswith("id")


def _note_llm_disabled(language: str) -> str:
    if _is_indonesian(language):
        return (
            "[Catatan] Fitur LLM dinonaktifkan di config.yaml "
            "(llm.enabled = false). Laporan di bawah ini dihasilkan "
            "menggunakan generator berbasis template bawaan (tanpa LLM)."
        )
    return (
        "[Note] LLM features are disabled in config.yaml "
        "(llm.enabled = false). The report below was generated using "
        "the built-in template engine only (no LLM)."
    )


def _note_llm_config_error(language: str, details: str) -> str:
    if _is_indonesian(language):
        return (
            "[Catatan] Laporan LLM tidak dapat dibuat karena masalah konfigurasi "
            "atau API key yang belum disetel. Sistem kembali ke laporan berbasis "
            "template.\n"
            f"Detail teknis: {details}"
        )
    return (
        "[Note] The LLM-based report could not be generated because of a "
        "configuration problem or missing API key. Falling back to the "
        "template-based report.\n"
        f"Technical details: {details}"
    )


def _note_llm_runtime_error(language: str, details: str) -> str:
    if _is_indonesian(language):
        return (
            "[Catatan] Terjadi kesalahan saat memanggil penyedia LLM. "
            "Sistem kembali ke laporan berbasis template.\n"
            f"Detail teknis: {details}"
        )
    return (
        "[Note] An error occurred while calling the LLM provider. Falling back "
        "to the template-based report.\n"
        f"Technical details: {details}"
    )


def generate_llm_report(
    metrics_bundle: Mapping[str, Any],
    config: Config,
    language: str = "en",
    *,
    mode_override: Optional[str] = None,
) -> str:
    """Generate a full insight report, optionally using an LLM.

    Parameters
    ----------
    metrics_bundle:
        The dictionary produced by :func:`analytics.build_metrics_bundle`.
    config:
        Loaded :class:`Config` object (includes :class:`LlmConfig`).
    language:
        Target output language for the report (e.g. ``"en"`` or ``"id"``).
    mode_override:
        Optional override for ``config.llm.mode``. One of ``"template"``,
        ``"llm"``, or ``"hybrid"``. If ``None``, the mode from the config
        object is used.

    Behaviour
    ---------
    - Always produces a valid report string.
    - If the effective mode is ``"template"``, the behaviour is identical to
      calling :func:`insights.generate_full_report`.
    - If the mode is ``"llm"`` or ``"hybrid"`` but LLM usage is disabled or
      misconfigured, the function returns the template report with a short
      explanatory note prepended.
    - If an unexpected error occurs when calling the LLM provider, the function
      also falls back to the template report with a note.
    """
    # Always build a template-based report as a safe baseline.
    template_report = insights.generate_full_report(metrics_bundle, language=language)

    llm_cfg = getattr(config, "llm", None)
    if llm_cfg is None:
        # Older configs or edge cases: behave like pure template mode.
        return template_report

    # Determine the effective mode for this call.
    mode = (mode_override or llm_cfg.mode or "template").strip().lower()
    if mode not in {"template", "llm", "hybrid"}:
        mode = "template"

    # If the caller explicitly wants the template engine, just return it.
    if mode == "template":
        return template_report

    # For "llm" and "hybrid" modes, we *attempt* to use the LLM. If anything
    # goes wrong, we fall back to the template-based report.
    # First, check whether LLM usage is enabled at all.
    if not llm_cfg.enabled:
        note = _note_llm_disabled(language)
        return f"{note}\n\n{template_report}"

    # Build prompts and call the LLM.
    opts = MetricsSummaryOptions(max_context_chars=llm_cfg.max_context_chars)
    system_prompt = build_system_prompt_for_report(language=language)
    template_for_hybrid = template_report if mode == "hybrid" else None
    user_prompt = build_user_prompt_for_report(
        metrics_bundle,
        language=language,
        template_report=template_for_hybrid,
        options=opts,
    )

    try:
        llm_text = llm_client.generate_text(
            user_prompt,
            config=config,
            system_prompt=system_prompt,
            language=language,
        )
        return llm_text.strip()
    except llm_client.LlmConfigError as exc:
        note = _note_llm_config_error(language, str(exc))
        return f"{note}\n\n{template_report}"
    except llm_client.LlmError as exc:
        note = _note_llm_runtime_error(language, str(exc))
        return f"{note}\n\n{template_report}"
"""LLM-powered insight generation for Financial Insight Generator.

This module provides a higher-level report generator that mirrors the existing
template-based ``generate_full_report`` function, but uses an LLM underneath
when enabled.

The main entry point is :func:`generate_llm_report`, which:

- Accepts the same ``metrics_bundle`` used by the template engine.
- Reads behaviour flags from ``config.llm`` (enabled, mode, etc.).
- Falls back gracefully to the template-based report if LLM usage is disabled
  or fails for any reason.
- (When possible) attaches a small RAG context built from vector-search over
  transactions so the LLM can ground its answer in concrete examples.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping, Optional

from .config import Config
from . import insights
from . import llm_client
from .llm_prompts import (
    MetricsSummaryOptions,
    build_system_prompt_for_report,
    build_user_prompt_for_report,
)
from .retrieval import retrieve_transactions_for_query

logger = logging.getLogger(__name__)


def _rag_debug_enabled() -> bool:
    """Return True if verbose RAG debugging is enabled via FIG_DEBUG_RAG."""
    value = os.environ.get("FIG_DEBUG_RAG", "")
    if not value:
        return False
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


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


def _build_rag_summary_for_report(
    metrics_bundle: Mapping[str, Any],
    config: Config,
    language: str,
) -> str:
    """Build a short, language-aware summary of retrieved transactions for RAG.

    This helper is intentionally conservative:

    - It only runs a single similarity search using a generic query derived
      from the current metrics bundle.
    - It limits itself to a handful of lines so it does not dominate the
      prompt budget.
    - If the vector store is disabled or an error occurs, it simply returns
      an empty string.
    """
    # Compose a lightweight query hint from the metrics bundle.
    overall = (metrics_bundle or {}).get("overall") or {}
    monthly_trend = (metrics_bundle or {}).get("monthly_trend") or {}
    anomaly = (metrics_bundle or {}).get("anomaly") or {}

    query_parts: list[str] = []

    total_revenue = overall.get("total_revenue")
    if isinstance(total_revenue, (int, float)):
        query_parts.append(f"overall revenue around {total_revenue:.2f}")

    n_transactions = overall.get("n_transactions")
    if isinstance(n_transactions, (int, float)):
        query_parts.append(f"{int(n_transactions)} transactions total")

    if monthly_trend.get("has_enough_data"):
        cur = monthly_trend.get("current_period")
        prev = monthly_trend.get("previous_period")
        if cur and prev:
            query_parts.append(f"month-over-month trend for {cur} vs {prev}")

    if anomaly.get("status") in {"high", "low"}:
        query_parts.append("recent daily revenue anomaly")

    if query_parts:
        query_text = (
            "transactions that best explain the following aspects of the data: "
            + "; ".join(str(p) for p in query_parts)
        )
    else:
        query_text = (
            "representative transactions including large amounts, key categories, "
            "and any recent anomalies"
        )

    filters: dict[str, Any] = {}
    if monthly_trend.get("current_period") is not None:
        filters["current_period"] = str(monthly_trend.get("current_period"))
    if anomaly.get("current_date") is not None:
        filters["anomaly_date"] = str(anomaly.get("current_date"))

    logger.debug(
        "FIG RAG: building report RAG context (query=%r, filters=%r)",
        query_text,
        filters,
    )

    # Ask the retrieval API for a small set of similar transactions.
    # Any configuration/runtime errors are swallowed (fail_silently=True).
    ctx = retrieve_transactions_for_query(
        query_text=query_text,
        config=config,
        top_k=None,
        language=language,
        filters=filters,
        fail_silently=True,
    )

    if ctx is None:
        logger.debug("FIG RAG: report RAG context not available (ctx=None)")
        return ""
    if not ctx.matches:
        logger.debug("FIG RAG: report RAG context has 0 matches")
        return ""

    # Build a compact, language-aware summary for the prompt.
    if _is_indonesian(language):
        header = "[Konteks RAG: contoh transaksi]"
    else:
        header = "[RAG context: sample transactions]"

    lines = [header]
    # We keep this intentionally short even if ctx.top_k is large.
    max_items = min(5, len(ctx.matches))
    for idx, match in enumerate(ctx.matches[:max_items], start=1):
        date = match.date or "?"
        category = match.category or "-"
        product = match.product or "-"
        amount = match.amount
        if isinstance(amount, (int, float)):
            amount_str = f"{amount:,.2f}"
        else:
            amount_str = "n/a"
        score_str = f"{match.score:.3f}"

        if _is_indonesian(language):
            line = (
                f"{idx}. {date} | kategori={category} | produk={product} | "
                f"jumlah={amount_str} | skor_kemiripan={score_str}"
            )
        else:
            line = (
                f"{idx}. {date} | category={category} | product={product} | "
                f"amount={amount_str} | similarity={score_str}"
            )
        lines.append(line)

    logger.debug(
        "FIG RAG: report RAG context built with %d matches",
        len(ctx.matches),
    )
    if _rag_debug_enabled():
        preview_lines = lines[: max_items + 1]
        print("[FIG RAG] Report retrieval context:")
        for line in preview_lines:
            print(f"[FIG RAG] {line}")

    # Hard cap to avoid overly long prompts.
    summary = "\n".join(lines)
    if len(summary) > 1200:
        summary = summary[:1200]

    return summary


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
    - When vector retrieval is available, a small RAG context (sample
      transactions) is appended to the user prompt so the LLM can ground its
      narrative in real examples.
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

    logger.debug("FIG LLM: generate_llm_report called (mode=%s)", mode)

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

    # Attach a compact RAG summary if available.
    rag_summary = _build_rag_summary_for_report(metrics_bundle, config, language)
    if rag_summary:
        user_prompt = f"{user_prompt}\n\n{rag_summary}"

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
        logger.warning("FIG LLM: configuration error in generate_llm_report: %s", exc)
        return f"{note}\n\n{template_report}"
    except llm_client.LlmError as exc:
        note = _note_llm_runtime_error(language, str(exc))
        logger.warning("FIG LLM: runtime error in generate_llm_report: %s", exc)
        return f"{note}\n\n{template_report}"
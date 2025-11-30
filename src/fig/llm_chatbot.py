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

import logging
import os
from typing import Any, Dict, Sequence

from . import llm_client
from .llm_prompts import (
    MetricsSummaryOptions,
    build_system_prompt_for_chat,
    build_user_prompt_for_question,
)
from .retrieval import retrieve_transactions_for_query

logger = logging.getLogger(__name__)


def _rag_debug_enabled() -> bool:
    """Return True if verbose RAG debugging is enabled via FIG_DEBUG_RAG."""
    value = os.environ.get("FIG_DEBUG_RAG", "")
    if not value:
        return False
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


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


def _build_rag_summary_for_question(
    question: str,
    metrics_bundle: Dict[str, Any] | None,
    config: Any,
    language: str,
) -> str:
    """Build a short summary of RAG results for a chat question.

    This helper is intentionally conservative:

    - It runs at most one vector search per question.
    - It limits itself to a handful of lines to avoid dominating the prompt.
    - If the vector store is disabled or an error occurs, it simply returns
    an empty string.
    """
    lang = _normalise_language(language)

    # Compose a simple retrieval query based on the question and overall metrics.
    overall = (metrics_bundle or {}).get("overall") or {}
    query_parts: list[str] = []

    if question:
        query_parts.append(str(question).strip())

    total_revenue = overall.get("total_revenue")
    if isinstance(total_revenue, (int, float)):
        query_parts.append(f"overall revenue around {total_revenue:.2f}")

    n_transactions = overall.get("n_transactions")
    if isinstance(n_transactions, (int, float)):
        query_parts.append(f"{int(n_transactions)} transactions total")

    if query_parts:
        query_text = (
            "transactions that help answer the following business question: "
            + "; ".join(query_parts)
        )
    else:
        query_text = (
            "transactions that are most relevant to recent revenue patterns "
            "and anomalies"
        )

    filters: Dict[str, Any] = {}
    if "date_range" in overall:
        filters["date_range"] = str(overall.get("date_range"))

    logger.debug(
        "FIG RAG: building chat RAG context (question=%r, query=%r, filters=%r)",
        question,
        query_text,
        filters,
    )

    ctx = retrieve_transactions_for_query(
        query_text=query_text,
        config=config,
        top_k=None,
        language=lang,
        filters=filters,
        fail_silently=True,
    )

    if ctx is None:
        logger.debug("FIG RAG: chat RAG context not available (ctx=None)")
        return ""
    if not ctx.matches:
        logger.debug("FIG RAG: chat RAG context has 0 matches")
        return ""

    if lang == "id":
        header = "[Konteks RAG: transaksi relevan]"
    else:
        header = "[RAG context: relevant transactions]"

    lines = [header]
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

        if lang == "id":
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
        "FIG RAG: chat RAG context built with %d matches",
        len(ctx.matches),
    )
    if _rag_debug_enabled():
        preview_lines = lines[: max_items + 1]
        print("[FIG RAG] Chat retrieval context:")
        for line in preview_lines:
            print(f"[FIG RAG] {line}")

    summary = "\n".join(lines)
    if len(summary) > 1200:
        summary = summary[:1200]
    return summary


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

    # If there is no LLM configuration or it is disabled, return a friendly note.
    if llm_cfg is None or not getattr(llm_cfg, "enabled", False):
        return _note_llm_config_error(
            language, "LLM features are disabled (llm.enabled = false)."
        )

    opts = MetricsSummaryOptions(
        max_context_chars=getattr(llm_cfg, "max_context_chars", 12_000)
    )
    system_prompt = build_system_prompt_for_chat(language=language)
    available_commands = _build_available_commands()
    user_prompt = build_user_prompt_for_question(
        question=question,
        metrics_bundle=metrics,
        language=language,
        available_commands=available_commands,
        options=opts,
    )

    # Attach a compact RAG summary if retrieval is available.
    rag_summary = _build_rag_summary_for_question(
        question=question,
        metrics_bundle=metrics,
        config=cfg,
        language=language,
    )

    # Expose the raw RAG summary in the mutable context for debugging / "show evidence" flows.
    context["last_rag_summary"] = rag_summary

    if rag_summary:
        user_prompt = f"{user_prompt}\n\n{rag_summary}"

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
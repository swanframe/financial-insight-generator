"""LangChain RAG chains for Financial Insight Generator.

This module defines LangChain Runnable pipelines (LCEL-style) that power
LLM-based features such as the financial insights report and interactive chat.

The main entry points are:

- ``build_report_chain`` – constructs a Runnable that accepts a dict input
  with at least ``{"metrics_bundle": ...}`` and returns a report string.
- ``generate_report_with_langchain`` – convenience helper that builds the
  report chain and invokes it in one step.
- ``build_chat_chain`` – constructs a Runnable that accepts a dict input with
  ``{"question": ..., "metrics_bundle": ...}`` and returns an answer string.
- ``answer_question_with_langchain`` – convenience helper that builds the
  chat chain and invokes it in one step.

These chains are designed to:

- Reuse existing prompt builders from :mod:`fig.llm_prompts`.
- Reuse the existing RAG layer via :mod:`fig.langchain_retriever`.
- Respect configuration, especially ``config.llm.max_context_chars``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from .config import Config
from .langchain_llm import get_langchain_chat_model
from .langchain_retriever import get_transactions_retriever
from .llm_prompts import (
    MetricsSummaryOptions,
    build_system_prompt_for_report,
    build_user_prompt_for_report,
    build_system_prompt_for_chat,
    build_user_prompt_for_question,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_indonesian(language: str | None) -> bool:
    """Return True if the language code looks like Indonesian."""
    if not language:
        return False
    return str(language).strip().lower().startswith("id")


def _build_report_rag_query(metrics_bundle: Mapping[str, Any]) -> str:
    """Build a generic similarity-search query from the metrics bundle.

    This does not need to be perfect; it just nudges retrieval toward
    transactions that are representative of the overall performance,
    recent trends and any anomalies.
    """
    overall = (metrics_bundle or {}).get("overall") or {}
    monthly_trend = (metrics_bundle or {}).get("monthly_trend") or {}
    anomaly = (metrics_bundle or {}).get("anomaly") or {}

    parts: list[str] = []

    total_revenue = overall.get("total_revenue")
    if isinstance(total_revenue, (int, float)):
        parts.append(f"overall revenue around {total_revenue:.2f}")

    n_transactions = overall.get("n_transactions")
    if isinstance(n_transactions, (int, float)):
        parts.append(f"{int(n_transactions)} total transactions")

    trend_label = monthly_trend.get("trend_label") or monthly_trend.get("trend_direction")
    if trend_label:
        parts.append(f"monthly trend is {trend_label}")

    has_anomaly = anomaly.get("has_anomaly") or anomaly.get("current_is_anomaly")
    if has_anomaly:
        parts.append("recent anomaly in revenue or orders")

    if parts:
        return "transactions that help explain: " + "; ".join(str(p) for p in parts)

    return (
        "representative transactions across main categories and recent periods, "
        "including high-value sales and any unusual changes"
    )


def format_rag_documents_for_report(
    docs: Sequence[Document],
    language: str,
    max_items: int = 5,
    hard_char_limit: int = 1_200,
) -> str:
    """Turn retrieved Documents into a compact RAG context block for reports.

    This is intentionally conservative: only a handful of lines are included
    and the final string is truncated to ``hard_char_limit`` characters.
    """
    if not docs:
        return ""

    lang_is_id = _is_indonesian(language)
    header = "[Konteks RAG: contoh transaksi]" if lang_is_id else "[RAG context: sample transactions]"

    lines = [header]

    for idx, doc in enumerate(docs[: max_items], start=1):
        meta = doc.metadata or {}
        date = meta.get("date") or "?"
        category = meta.get("category") or meta.get("category_name") or "-"
        product = meta.get("product") or meta.get("product_name") or "-"
        amount = meta.get("amount")
        score = meta.get("score")

        if isinstance(amount, (int, float)):
            amount_str = f"{amount:,.2f}"
        else:
            amount_str = "n/a"

        if isinstance(score, (int, float)):
            score_str = f"{float(score):.3f}"
        else:
            score_str = "n/a"

        if lang_is_id:
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

    summary = "\n".join(lines)
    if len(summary) > hard_char_limit:
        summary = summary[:hard_char_limit] + "\n[Context truncated for length]"
    return summary


def format_rag_documents_for_chat(
    docs: Sequence[Document],
    language: str,
    max_items: int = 5,
    hard_char_limit: int = 1_200,
) -> str:
    """Turn retrieved Documents into a compact RAG context block for chat.

    The formatting mirrors the native ``_build_rag_summary_for_question`` helper
    in :mod:`fig.llm_chatbot`, but operates on LangChain Documents instead of
    :class:`RetrievedTransaction` objects.
    """
    if not docs:
        return ""

    lang_is_id = _is_indonesian(language)
    header = "[Konteks RAG: transaksi relevan]" if lang_is_id else "[RAG context: relevant transactions]"

    lines = [header]

    for idx, doc in enumerate(docs[: max_items], start=1):
        meta = doc.metadata or {}
        date = meta.get("date") or "?"
        category = meta.get("category") or meta.get("category_name") or "-"
        product = meta.get("product") or meta.get("product_name") or "-"
        amount = meta.get("amount")
        score = meta.get("score")

        if isinstance(amount, (int, float)):
            amount_str = f"{amount:,.2f}"
        else:
            amount_str = "n/a"

        if isinstance(score, (int, float)):
            score_str = f"{float(score):.3f}"
        else:
            score_str = "n/a"

        if lang_is_id:
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

    summary = "\n".join(lines)
    if len(summary) > hard_char_limit:
        summary = summary[:hard_char_limit] + "\n[Context truncated for length]"
    return summary


# ---------------------------------------------------------------------------
# Report chain
# ---------------------------------------------------------------------------


def build_report_chain(
    config: Config,
    language: Optional[str] = None,
):
    """Build a LangChain Runnable that generates an LLM-powered report.

    The returned Runnable expects an input mapping with at least:

        {
            "metrics_bundle": metrics_bundle,          # required
            "template_report": optional_template_str,  # optional (for 'hybrid' mode)
        }

    and returns a final report string.

    Parameters
    ----------
    config:
        Global FIG configuration.

    language:
        Optional override for the target language. When not provided,
        ``config.ui.language`` is used.
    """
    effective_language = language or config.ui.language
    max_chars = int(config.llm.max_context_chars or 12_000)

    # Underlying chat model.
    chat_model = get_langchain_chat_model(config)

    # Prompt template: we feed fully-rendered strings as variables.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("user", "{user_prompt}"),
        ]
    )

    # Retriever (wraps the existing RAG layer).
    retriever = get_transactions_retriever(
        config,
        language=effective_language,
        top_k=config.vector_store.default_top_k,
        fail_silently=True,
    )

    def prepare_prompt_components(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare system prompt and base user prompt from the metrics bundle."""
        metrics_bundle = inputs["metrics_bundle"]
        template_report = inputs.get("template_report")

        opts = MetricsSummaryOptions(
            max_context_chars=max_chars,
            # We keep the other defaults (time series rows, segment counts).
        )

        system_prompt = build_system_prompt_for_report(language=effective_language)
        user_prompt_base = build_user_prompt_for_report(
            metrics_bundle,
            language=effective_language,
            template_report=template_report,
            options=opts,
        )

        return {
            "metrics_bundle": metrics_bundle,
            "template_report": template_report,
            "system_prompt": system_prompt,
            "user_prompt_base": user_prompt_base,
        }

    def compute_rag_context(state: Dict[str, Any]) -> str:
        """Compute an optional RAG context block based on the metrics bundle."""
        # If the vector store is disabled, we skip retrieval entirely.
        if not config.vector_store.enabled:
            return ""

        metrics_bundle = state["metrics_bundle"]
        query_text = _build_report_rag_query(metrics_bundle)
        docs = retriever.invoke(query_text)
        return format_rag_documents_for_report(docs, language=effective_language)

    def build_prompt_inputs(state: Dict[str, Any]) -> Dict[str, str]:
        """Combine base user prompt and RAG context into final prompt strings."""
        system_prompt: str = state["system_prompt"]
        user_prompt_base: str = state["user_prompt_base"]
        rag_context: str = state.get("rag_context", "")

        # Attach RAG context at the end of the user prompt when available.
        if rag_context:
            user_prompt = f"{user_prompt_base}\n\n{rag_context}"
        else:
            user_prompt = user_prompt_base

        # Hard cap on user prompt length for safety.
        if len(user_prompt) > max_chars:
            user_prompt = user_prompt[:max_chars] + "\n\n[Context truncated for length]"

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

    # LCEL pipeline.
    chain = (
        RunnableLambda(prepare_prompt_components)
        | {
            # Pass through system + base prompt, and compute RAG context in parallel.
            "system_prompt": lambda s: s["system_prompt"],
            "user_prompt_base": lambda s: s["user_prompt_base"],
            "rag_context": lambda s: compute_rag_context(s),
        }
        | RunnableLambda(build_prompt_inputs)
        | prompt
        | chat_model
        | StrOutputParser()
    )

    return chain


def generate_report_with_langchain(
    metrics_bundle: Mapping[str, Any],
    config: Config,
    language: Optional[str] = None,
    template_report: Optional[str] = None,
) -> str:
    """Generate a financial insights report using the LangChain report chain.

    This is a thin convenience wrapper around :func:`build_report_chain`. It
    mirrors the shape of :func:`fig.llm_insights.generate_llm_report` so that
    the CLI / orchestration layer can easily switch between the native and
    LangChain engines.
    """
    chain = build_report_chain(config=config, language=language)
    return chain.invoke(
        {
            "metrics_bundle": metrics_bundle,
            "template_report": template_report,
        }
    )


# ---------------------------------------------------------------------------
# Chat / Q&A chain
# ---------------------------------------------------------------------------

# Default list of CLI-style commands that the assistant may refer to in answers.
_AVAILABLE_CHAT_COMMANDS: Sequence[str] = (
    "summary",
    "overview",
    "top categories",
    "top products",
    "top customers",
    "top channels",
    "trend",
    "anomaly",
    "time series",
)


def build_chat_chain(
    config: Config,
    language: Optional[str] = None,
):
    """Build a LangChain Runnable that answers ad-hoc questions.

    The returned Runnable expects an input mapping with at least:

        {
            "question": question_str,
            "metrics_bundle": metrics_bundle,
        }

    and returns an answer string.
    """
    effective_language = language or config.ui.language
    max_chars = int(config.llm.max_context_chars or 12_000)

    chat_model = get_langchain_chat_model(config)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("user", "{user_prompt}"),
        ]
    )

    retriever = get_transactions_retriever(
        config,
        language=effective_language,
        top_k=config.vector_store.default_top_k,
        fail_silently=True,
    )

    def prepare_chat_components(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare system and user prompts for the chat question."""
        question = str(inputs["question"])
        metrics_bundle = inputs.get("metrics_bundle") or {}

        opts = MetricsSummaryOptions(
            max_context_chars=max_chars,
        )

        system_prompt = build_system_prompt_for_chat(language=effective_language)
        user_prompt_base = build_user_prompt_for_question(
            question=question,
            metrics_bundle=metrics_bundle,
            language=effective_language,
            available_commands=_AVAILABLE_CHAT_COMMANDS,
            options=opts,
        )

        return {
            "question": question,
            "metrics_bundle": metrics_bundle,
            "system_prompt": system_prompt,
            "user_prompt_base": user_prompt_base,
        }

    def compute_chat_rag_context(state: Dict[str, Any]) -> str:
        """Compute an optional RAG context block for the chat question."""
        if not config.vector_store.enabled:
            return ""

        question = state["question"]
        # For chat we keep the retrieval query simple: the question itself is
        # usually a good semantic signal; the retriever already sees rich
        # metadata in the index.
        query_text = str(question).strip() or "sales transactions relevant to this question"
        docs = retriever.invoke(query_text)
        return format_rag_documents_for_chat(docs, language=effective_language)

    def build_chat_prompt_inputs(state: Dict[str, Any]) -> Dict[str, str]:
        """Combine base user prompt and RAG context into final prompt strings."""
        system_prompt: str = state["system_prompt"]
        user_prompt_base: str = state["user_prompt_base"]
        rag_context: str = state.get("rag_context", "")

        if rag_context:
            user_prompt = f"{user_prompt_base}\n\n{rag_context}"
        else:
            user_prompt = user_prompt_base

        if len(user_prompt) > max_chars:
            user_prompt = user_prompt[:max_chars] + "\n\n[Context truncated for length]"

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

    chain = (
        RunnableLambda(prepare_chat_components)
        | {
            "system_prompt": lambda s: s["system_prompt"],
            "user_prompt_base": lambda s: s["user_prompt_base"],
            "rag_context": lambda s: compute_chat_rag_context(s),
        }
        | RunnableLambda(build_chat_prompt_inputs)
        | prompt
        | chat_model
        | StrOutputParser()
    )

    return chain


def answer_question_with_langchain(
    question: str,
    metrics_bundle: Mapping[str, Any],
    config: Config,
    language: Optional[str] = None,
) -> str:
    """Answer a question using the LangChain chat chain.

    This mirrors :func:`fig.llm_chatbot.answer_question` but routes the logic
    through the LangChain pipeline instead of the native llm_client path.
    """
    chain = build_chat_chain(config=config, language=language)
    return chain.invoke(
        {
            "question": question,
            "metrics_bundle": metrics_bundle,
        }
    )
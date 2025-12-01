"""Simple chat-like interface for Financial Insight Generator.

This module implements a small REPL on top of the analytics / insights layer.
It supports a mix of:

- Rule-based commands (e.g. "/summary", "/trend").
- Free-form questions answered by an LLM (optionally with RAG).

The actual LLM answering logic lives in:

- fig.llm_chatbot (native / non-LangChain path).
- fig.langchain_chains.answer_question_with_langchain (LangChain path).
"""

from __future__ import annotations

from typing import Any, Dict

from . import llm_chatbot
from .langchain_chains import answer_question_with_langchain


def _print_banner() -> None:
    print("------------------------------------------------------------")
    print("Interactive assistant – ask questions about your data.")
    print("Type '/help' for commands, '/quit' to exit.")
    print("------------------------------------------------------------")


def _print_help() -> None:
    print()
    print("Available commands:")
    print("  /summary       - Show a high-level summary of key metrics.")
    print("  /trend         - Describe recent revenue trends.")
    print("  /help          - Show this help message.")
    print("  /quit or /exit - Leave the assistant.")
    print()
    print("Anything else will be treated as a free-form question.")
    print()


def start_chat_interface(context: Dict[str, Any]) -> None:
    """Start the interactive chat loop.

    Parameters
    ----------
    context:
        Dictionary containing shared objects from the main pipeline, typically:

        - ``df``: cleaned transaction DataFrame.
        - ``metrics``: metrics bundle from fig.analytics.build_metrics_bundle.
        - ``config``: loaded Config object.
        - ``language``: effective UI language (e.g. "en", "id").
        - ``engine``: orchestration engine ("native" or "langchain").
    """
    _print_banner()
    _print_help()

    cfg = context.get("config")
    llm_cfg = getattr(cfg, "llm", None) if cfg is not None else None

    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):  # pragma: no cover - interactive only
            print()
            print("Exiting chat.")
            break

        stripped = user_input.strip()

        if not stripped:
            continue

        lower = stripped.lower()

        if lower in {"/quit", "/exit"}:
            print("Goodbye!")
            break
        elif lower == "/help":
            _print_help()
            continue
        elif lower == "/summary":
            # Prefer a dedicated helper if it exists; otherwise fall back to
            # a generic free-form question.
            if hasattr(llm_chatbot, "build_summary_answer"):
                summary = llm_chatbot.build_summary_answer(context)
            else:
                summary = llm_chatbot.answer_freeform_question(
                    "Give me a high-level summary of the key business metrics.",
                    context,
                )
            print()
            print(summary)
            print()
            continue

        elif lower == "/trend":
            if hasattr(llm_chatbot, "build_trend_answer"):
                trend = llm_chatbot.build_trend_answer(context)
            else:
                trend = llm_chatbot.answer_freeform_question(
                    "Describe the recent revenue trend over time.",
                    context,
                )
            print()
            print(trend)
            print()
            continue

        # ------------------------------------------------------------------
        # Free-form question: choose between native and LangChain engines.
        # ------------------------------------------------------------------
        engine = str(context.get("engine", "native") or "native").lower()
        llm_enabled = bool(getattr(llm_cfg, "enabled", False))
        llm_mode = getattr(llm_cfg, "mode", "template") if llm_cfg is not None else "template"

        use_langchain = (
            engine == "langchain"
            and llm_enabled
            and llm_mode in {"llm", "hybrid"}
        )

        if use_langchain:
            metrics_bundle = context.get("metrics") or {}
            language = str(context.get("language") or "en")

            try:
                answer = answer_question_with_langchain(
                    question=user_input,
                    config=cfg,
                    metrics_bundle=metrics_bundle,
                    language=language,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                print()
                print(
                    f"[ERROR] LangChain chat failed, falling back to native engine: {exc}"
                )
                print()
                answer = llm_chatbot.answer_freeform_question(user_input, context)
        else:
            # Legacy/native path using fig.llm_chatbot.
            answer = llm_chatbot.answer_freeform_question(user_input, context)

        print()
        print(answer)
        print()
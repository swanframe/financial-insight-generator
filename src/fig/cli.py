"""Command-line interface for Financial Insight Generator.

This module provides a convenient entry point for:

- Running the full data → analytics → insights pipeline.
- Generating either a template-based or LLM-based report.
- Entering an interactive chat mode on top of the computed metrics.

The CLI intentionally stays thin and delegates to the core modules:

- fig.preprocessing.load_and_clean_transactions
- fig.analytics.build_metrics_bundle
- fig.insights.generate_full_report
- fig.llm_insights.generate_llm_report
- fig.chatbot.start_chat_interface
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from .preprocessing import load_and_clean_transactions
from .analytics import build_metrics_bundle
from .insights import generate_full_report
from .llm_insights import generate_llm_report
from .i18n import get_translator
from . import chatbot
from .langchain_chains import generate_report_with_langchain


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the FIG CLI."""
    parser = argparse.ArgumentParser(
        description="Financial Insight Generator (FIG) CLI",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip printing the report (useful if you only want interactive chat).",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enter interactive chat mode after generating the report.",
    )
    parser.add_argument(
        "-l",
        "--lang",
        help="Override UI/report language (e.g. 'en', 'id'). Defaults to config.ui.language.",
    )
    parser.add_argument(
        "--report-mode",
        choices=["template", "llm", "hybrid"],
        help=(
            "Report generation mode override. "
            "Defaults to config.llm.mode if not provided."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=["native", "langchain"],
        help=(
            "Orchestration engine for LLM-powered features. "
            "Defaults to config.orchestration.engine."
        ),
    )
    return parser.parse_args(argv)


def _resolve_language(args: argparse.Namespace, cfg: Any) -> str:
    """Determine the effective UI language."""
    config_lang = getattr(getattr(cfg, "ui", None), "language", "en") or "en"
    raw_lang = args.lang or config_lang
    return str(raw_lang).strip().lower() if raw_lang else "en"


def _resolve_engine(args: argparse.Namespace, cfg: Any) -> str:
    """Determine which orchestration engine to use ('native' or 'langchain')."""
    cfg_orch = getattr(cfg, "orchestration", None)
    cfg_engine = getattr(cfg_orch, "engine", "native") if cfg_orch is not None else "native"
    raw_engine = args.engine or cfg_engine or "native"
    engine = str(raw_engine).strip().lower()
    if engine not in {"native", "langchain"}:
        engine = "native"
    return engine


def _resolve_report_mode(args: argparse.Namespace, cfg: Any) -> str:
    """Determine the report mode ('template', 'llm', or 'hybrid')."""
    llm_cfg = getattr(cfg, "llm", None)
    cfg_mode = getattr(llm_cfg, "mode", "template") if llm_cfg is not None else "template"
    cfg_mode = cfg_mode or "template"

    report_mode = args.report_mode or cfg_mode
    if report_mode not in {"template", "llm", "hybrid"}:
        report_mode = "template"
    return report_mode


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the FIG CLI."""
    args = parse_args(argv)
    config_path = Path(args.config)

    # Run the preprocessing pipeline and load the cleaned DataFrame + config.
    df, pipeline_report, cfg = load_and_clean_transactions(config_path)

    language = _resolve_language(args, cfg)
    t = get_translator(language)

    engine = _resolve_engine(args, cfg)
    report_mode = _resolve_report_mode(args, cfg)

    llm_cfg = getattr(cfg, "llm", None)
    llm_enabled: bool = bool(getattr(llm_cfg, "enabled", False))

    # Build analytics / metrics once and reuse for report + chat.
    metrics_bundle = build_metrics_bundle(df, cfg)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    if not args.no_report:
        print()
        print(t("cli.banner"))
        print()

        if (not llm_enabled) or report_mode == "template":
            # Pure template path (LLM disabled or mode explicitly set to "template").
            report_text = generate_full_report(metrics_bundle, language=language)
        else:
            # LLM-powered modes ("llm" or "hybrid").
            if engine == "langchain":
                # For hybrid mode, generate the template report and pass it as context.
                template_report: Optional[str] = None
                if report_mode == "hybrid":
                    template_report = generate_full_report(metrics_bundle, language=language)

                report_text = generate_report_with_langchain(
                    metrics_bundle=metrics_bundle,
                    config=cfg,
                    language=language,
                    template_report=template_report,
                )
            else:
                # Legacy/native path using fig.llm_insights.
                report_text = generate_llm_report(
                    metrics_bundle,
                    cfg,
                    language=language,
                    mode_override=report_mode,
                )

        print(report_text)

    # ------------------------------------------------------------------
    # Interactive chat
    # ------------------------------------------------------------------
    if args.interactive:
        context: Dict[str, Any] = {
            "df": df,
            "metrics": metrics_bundle,
            "config": cfg,
            "language": language,
            "engine": engine,
        }
        chatbot.start_chat_interface(context)


if __name__ == "__main__":  # pragma: no cover - manual CLI entry
    main()
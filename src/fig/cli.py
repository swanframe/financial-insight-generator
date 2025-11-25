"""Command-line interface for Financial Insight Generator.

This module provides a simple entry point to:
- Run the end-to-end pipeline (config -> data -> analytics -> insights)
- Print a full financial insight report
- Optionally start an interactive chat-like loop
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .preprocessing import load_and_clean_transactions
from .analytics import build_metrics_bundle
from .insights import generate_full_report
from . import chatbot
from .i18n import get_translator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Financial Insight Generator (FIG) CLI",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",  # boolean flag
        help="Do not print the full insight report on startup.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start interactive chat-like mode after initial analysis.",
    )
    parser.add_argument(
        "-l",
        "--lang",
        type=str,
        help=(
            "Language for user-facing output (e.g., 'en' or 'id'). "
            "Defaults to ui.language in config.yaml if not provided."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    args = parse_args(argv)

    config_path = Path(args.config)

    # --- Run the data pipeline ---
    df, pipeline_report, cfg = load_and_clean_transactions(config_path)

    # --- Determine effective language: CLI flag > config > default 'en' ---
    if hasattr(cfg, "ui"):
        config_lang = getattr(cfg.ui, "language", "en") or "en"
    else:
        config_lang = "en"

    raw_lang = args.lang or config_lang
    language = str(raw_lang).strip().lower() if raw_lang else "en"
    t = get_translator(language)

    # --- Build metrics bundle ---
    metrics_bundle = build_metrics_bundle(df, cfg)

    # --- Print full report unless suppressed ---
    if not args.no_report:
        print()
        print(t("cli.banner"))
        print()
        report_text = generate_full_report(metrics_bundle, language=language)
        print(report_text)

    # --- Optionally start interactive mode ---
    if args.interactive:
        context = {
            "df": df,
            "metrics": metrics_bundle,
            "config": cfg,
            "language": language,
        }
        chatbot.start_chat_interface(context)


if __name__ == "__main__":
    main()
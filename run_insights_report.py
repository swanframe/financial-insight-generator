"""Convenience script to generate a full financial insight report.

This script runs the data pipeline + analytics, then prints a report to stdout.
It supports both the original template-based report and the LLM-powered report,
depending on configuration and command-line flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from fig.preprocessing import load_and_clean_transactions
from fig.analytics import build_metrics_bundle
from fig.insights import generate_full_report
from fig.llm_insights import generate_llm_report
from fig.i18n import get_translator


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a financial insight report from config + data.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "-l",
        "--lang",
        type=str,
        help=(
            "Language for the report (e.g., 'en' or 'id'). "
            "Defaults to ui.language in config.yaml if not provided."
        ),
    )
    parser.add_argument(
        "--report-mode",
        choices=["template", "llm", "hybrid"],
        help=(
            "Override llm.mode from config.yaml for this run. "
            "Use 'template' for the original template-based report, "
            "'llm' for an LLM-generated narrative, or 'hybrid' to let "
            "the LLM refine the template report."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    config_path = Path(args.config)
    df, pipeline_report, cfg = load_and_clean_transactions(config_path)

    # Determine effective language.
    if hasattr(cfg, "ui"):
        config_lang = getattr(cfg.ui, "language", "en") or "en"
    else:
        config_lang = "en"
    raw_lang = args.lang or config_lang
    language = str(raw_lang).strip().lower() if raw_lang else "en"
    t = get_translator(language)

    metrics_bundle = build_metrics_bundle(df, cfg)

    # Determine effective report mode.
    if hasattr(cfg, "llm"):
        cfg_mode = getattr(cfg.llm, "mode", "template") or "template"
    else:
        cfg_mode = "template"
    report_mode = args.report_mode or cfg_mode
    if report_mode not in {"template", "llm", "hybrid"}:
        report_mode = "template"

    print()
    print(t("cli.banner"))
    print()

    if report_mode == "template":
        report_text = generate_full_report(metrics_bundle, language=language)
    else:
        report_text = generate_llm_report(
            metrics_bundle,
            cfg,
            language=language,
            mode_override=report_mode,
        )

    print(report_text)


if __name__ == "__main__":
    main()
"""Convenience script to run the FIG pipeline and print a single report.

This script is a thin wrapper around the core FIG modules. It:

1. Loads and cleans transaction data using the configuration file.
2. Computes analytics / metrics.
3. Generates either:
   - a template-based report, or
   - an LLM-based report (optionally with hybrid template context).
4. Prints the report to stdout and optionally writes it to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from src.fig.preprocessing import load_and_clean_transactions
from src.fig.analytics import build_metrics_bundle
from src.fig.insights import generate_full_report
from src.fig.llm_insights import generate_llm_report
from src.fig.langchain_chains import generate_report_with_langchain
from src.fig.i18n import get_translator


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the one-shot report script."""
    parser = argparse.ArgumentParser(
        description="Run the Financial Insight Generator and print a report.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
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
            "Orchestration engine for LLM-powered report generation. "
            "Defaults to config.orchestration.engine."
        ),
    )
    return parser.parse_args(argv)


def _resolve_language(args: argparse.Namespace, cfg: object) -> str:
    """Determine the effective UI language."""
    ui_cfg = getattr(cfg, "ui", None)
    config_lang = getattr(ui_cfg, "language", "en") if ui_cfg is not None else "en"
    config_lang = config_lang or "en"
    raw_lang = args.lang or config_lang
    return str(raw_lang).strip().lower() if raw_lang else "en"


def _resolve_engine(args: argparse.Namespace, cfg: object) -> str:
    """Determine which orchestration engine to use ('native' or 'langchain')."""
    orch_cfg = getattr(cfg, "orchestration", None)
    cfg_engine = getattr(orch_cfg, "engine", "native") if orch_cfg is not None else "native"
    cfg_engine = cfg_engine or "native"
    raw_engine = args.engine or cfg_engine
    engine = str(raw_engine).strip().lower() if raw_engine else "native"
    if engine not in {"native", "langchain"}:
        engine = "native"
    return engine


def _resolve_report_mode(args: argparse.Namespace, cfg: object) -> str:
    """Determine the report mode ('template', 'llm', or 'hybrid')."""
    llm_cfg = getattr(cfg, "llm", None)
    cfg_mode = getattr(llm_cfg, "mode", "template") if llm_cfg is not None else "template"
    cfg_mode = cfg_mode or "template"

    report_mode = args.report_mode or cfg_mode
    if report_mode not in {"template", "llm", "hybrid"}:
        report_mode = "template"
    return report_mode


def main(argv: Optional[List[str]] = None) -> None:
    """Run the FIG pipeline and print a single report."""
    args = parse_args(argv)
    config_path = Path(args.config)

    # Preprocessing pipeline: returns cleaned DataFrame, pipeline report, and config.
    df, pipeline_report, cfg = load_and_clean_transactions(config_path)

    language = _resolve_language(args, cfg)
    t = get_translator(language)

    engine = _resolve_engine(args, cfg)
    report_mode = _resolve_report_mode(args, cfg)

    llm_cfg = getattr(cfg, "llm", None)
    llm_enabled: bool = bool(getattr(llm_cfg, "enabled", False))

    # Build analytics / metrics.
    metrics_bundle = build_metrics_bundle(df, cfg)

    print(t("cli.banner"))
    print()

    if (not llm_enabled) or report_mode == "template":
        # Pure template path when LLM is disabled or mode is "template".
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

    # Optionally save the report to disk based on config.output.
    output_cfg = getattr(cfg, "output", None)
    save_report = bool(getattr(output_cfg, "save_report", False)) if output_cfg else False
    if save_report and output_cfg is not None:
        report_path = Path(output_cfg.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
        print()
        print(t("cli.report_saved_to").format(path=report_path))


if __name__ == "__main__":  # pragma: no cover - manual script entry
    main()
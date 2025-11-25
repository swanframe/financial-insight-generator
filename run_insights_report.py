"""Generate a full financial insight report for the sample dataset."""

from fig.preprocessing import load_and_clean_transactions
from fig.analytics import build_metrics_bundle
from fig.insights import generate_full_report


def main() -> None:
    df, report, cfg = load_and_clean_transactions("config.yaml")

    metrics_bundle = build_metrics_bundle(df, cfg)

    # Determine language from config if available
    language = "en"
    if hasattr(cfg, "ui"):
        ui_language = getattr(cfg.ui, "language", "en")
        if ui_language:
            language = str(ui_language).strip().lower()

    report_text = generate_full_report(metrics_bundle, language=language)

    print(report_text)


if __name__ == "__main__":
    main()
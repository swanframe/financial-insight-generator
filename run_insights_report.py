"""Generate a full financial insight report for the sample dataset."""

from fig.preprocessing import load_and_clean_transactions
from fig.analytics import build_metrics_bundle
from fig.insights import generate_full_report


def main() -> None:
    df, report, cfg = load_and_clean_transactions("config.yaml")

    metrics_bundle = build_metrics_bundle(df, cfg)
    report_text = generate_full_report(metrics_bundle)

    print(report_text)


if __name__ == "__main__":
    main()
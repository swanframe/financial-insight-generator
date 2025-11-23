"""Small helper script to test the data pipeline on the sample dataset."""

from fig.preprocessing import load_and_clean_transactions


def main() -> None:
    df, report, cfg = load_and_clean_transactions("config.yaml")

    print("\n=== Validation Report ===")
    for k, v in report["validation"].items():
        print(f"{k}: {v}")

    print("\n=== Cleaning Summary ===")
    for k, v in report["cleaning"].items():
        print(f"{k}: {v}")

    print("\n=== Sample of Cleaned Data ===")
    print(df.head())


if __name__ == "__main__":
    main()
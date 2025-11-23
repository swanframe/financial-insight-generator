from pathlib import Path

import pytest

from fig.config import load_config
from fig.preprocessing import load_and_clean_transactions
from fig.analytics import build_metrics_bundle


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Path to the project root (where config.yaml lives)."""
    return Path(".").resolve()


@pytest.fixture(scope="session")
def config_obj(project_root: Path):
    """Loaded Config object from config.yaml."""
    cfg_path = project_root / "config.yaml"
    return load_config(cfg_path)


@pytest.fixture(scope="session")
def clean_df_and_report(config_obj):
    """Run the full data pipeline once and reuse for tests."""
    # We rely on config.yaml paths, so just pass the path.
    df, report, cfg = load_and_clean_transactions("config.yaml")
    return df, report, cfg


@pytest.fixture(scope="session")
def clean_df(clean_df_and_report):
    df, _, _ = clean_df_and_report
    return df


@pytest.fixture(scope="session")
def metrics_bundle(clean_df, config_obj):
    """Full metrics bundle from the cleaned data."""
    return build_metrics_bundle(clean_df, config_obj)
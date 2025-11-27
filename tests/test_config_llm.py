from pathlib import Path

from fig.config import load_config


def test_load_config_llm_defaults_when_section_missing(tmp_path):
    """If llm section is missing, defaults should be applied and LLM disabled."""
    cfg_path = tmp_path / "minimal_config.yaml"
    cfg_path.write_text(
        """
data:
  input_path: "data/raw/sample.csv"
columns:
  date: "order_date"
  amount: "amount"
        """,
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    # LLM should be present but disabled with sensible defaults.
    assert cfg.llm.enabled is False
    assert cfg.llm.provider == "openai"
    assert cfg.llm.model  # non-empty string
    assert cfg.llm.mode == "template"


def test_load_config_llm_invalid_mode_defaults_to_template(tmp_path):
    """Unknown llm.mode values should be normalised to 'template'."""
    cfg_path = tmp_path / "config_invalid_mode.yaml"
    cfg_path.write_text(
        """
data:
  input_path: "data/raw/sample.csv"
columns:
  date: "order_date"
  amount: "amount"
llm:
  enabled: true
  mode: "not-a-real-mode"
        """,
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.llm.enabled is True
    # Invalid mode should be corrected to 'template'
    assert cfg.llm.mode == "template"
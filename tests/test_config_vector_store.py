from pathlib import Path

from fig.config import load_config


def test_embeddings_defaults_derived_from_llm(tmp_path):
    """If embeddings section is missing, defaults should be derived from the LLM config."""
    cfg_path = tmp_path / "config_embeddings_missing.yaml"
    cfg_path.write_text(
        """
data:
  input_path: "data/raw/sample.csv"
columns:
  date: "order_date"
  amount: "amount"
llm:
  enabled: true
  provider: "openai"
  model: "gpt-4.1-mini"
  api_key_env_var: "FIG_TEST_LLM_KEY"
  timeout_seconds: 15
        """,
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    # LLM should be wired as requested
    assert cfg.llm.enabled is True
    assert cfg.llm.provider == "openai"
    assert cfg.llm.api_key_env_var == "FIG_TEST_LLM_KEY"
    assert cfg.llm.timeout_seconds == 15

    # Embeddings should derive provider, api key env var, and timeout from LLM
    assert cfg.embeddings.provider == cfg.llm.provider
    assert cfg.embeddings.api_key_env_var == cfg.llm.api_key_env_var
    assert cfg.embeddings.timeout_seconds == cfg.llm.timeout_seconds
    # Model falls back to the embeddings default, not the chat model
    assert cfg.embeddings.model == "text-embedding-3-small"


def test_embeddings_section_can_override_llm_defaults(tmp_path):
    """Explicit embeddings section should override derived defaults."""
    cfg_path = tmp_path / "config_embeddings_custom.yaml"
    cfg_path.write_text(
        """
data:
  input_path: "data/raw/sample.csv"
columns:
  date: "order_date"
  amount: "amount"
llm:
  enabled: true
  provider: "openai"
  api_key_env_var: "FIG_TEST_LLM_KEY"
embeddings:
  provider: "custom-provider"
  model: "my-embedding-model"
  api_key_env_var: "FIG_TEST_EMBED_KEY"
  timeout_seconds: 42
        """,
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.embeddings.provider == "custom-provider"
    assert cfg.embeddings.model == "my-embedding-model"
    assert cfg.embeddings.api_key_env_var == "FIG_TEST_EMBED_KEY"
    assert cfg.embeddings.timeout_seconds == 42


def test_vector_store_defaults_when_section_missing(tmp_path):
    """If vector_store section is missing, defaults should be applied."""
    cfg_path = tmp_path / "config_vector_missing.yaml"
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

    assert cfg.vector_store.enabled is False
    assert cfg.vector_store.provider == "chroma"
    assert cfg.vector_store.persist_path == Path("data/vector_store")
    assert cfg.vector_store.collection_name == "fig_transactions"
    assert cfg.vector_store.default_top_k == 5


def test_vector_store_custom_values(tmp_path):
    """vector_store section should be parsed with sensible types."""
    cfg_path = tmp_path / "config_vector_custom.yaml"
    cfg_path.write_text(
        """
data:
  input_path: "data/raw/sample.csv"
columns:
  date: "order_date"
  amount: "amount"
vector_store:
  enabled: true
  provider: "in_memory"
  persist_path: "tmp/vectors"
  collection_name: "custom_collection"
  default_top_k: 10
        """,
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.vector_store.enabled is True
    assert cfg.vector_store.provider == "in_memory"
    assert isinstance(cfg.vector_store.persist_path, Path)
    assert cfg.vector_store.persist_path.as_posix().endswith("tmp/vectors")
    assert cfg.vector_store.collection_name == "custom_collection"
    assert cfg.vector_store.default_top_k == 10
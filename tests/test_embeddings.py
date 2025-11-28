import pytest

from fig import embeddings
from fig.config import load_config
from fig.config import Config


def _load_minimal_config(tmp_path) -> Config:
    cfg_path = tmp_path / "config_minimal.yaml"
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
        """,
        encoding="utf-8",
    )
    return load_config(cfg_path)


def test_embed_texts_raises_config_error_when_api_key_missing(monkeypatch, tmp_path):
    """If embeddings provider is OpenAI but the env var is missing, raise EmbeddingsConfigError early."""
    cfg = _load_minimal_config(tmp_path)
    # Ensure embeddings use OpenAI and look up a specific env var.
    cfg.embeddings.provider = "openai"
    cfg.embeddings.api_key_env_var = "FIG_TEST_MISSING_EMBED_KEY"

    # Ensure the env var really isn't set.
    monkeypatch.delenv("FIG_TEST_MISSING_EMBED_KEY", raising=False)

    with pytest.raises(embeddings.EmbeddingsConfigError) as excinfo:
        embeddings.embed_texts(["Hello"], config=cfg)

    msg = str(excinfo.value)
    assert "FIG_TEST_MISSING_EMBED_KEY" in msg
    assert "not set or empty" in msg


def test_embed_texts_unsupported_provider(monkeypatch, tmp_path):
    """Selecting an unsupported provider should raise a clear configuration error."""
    cfg = _load_minimal_config(tmp_path)

    # Ensure the configured API key env var exists so we pass the key check
    # for providers that require it.
    monkeypatch.setenv(cfg.embeddings.api_key_env_var, "dummy-key")

    # Use an unsupported provider name.
    cfg.embeddings.provider = "unsupported-provider"

    with pytest.raises(embeddings.EmbeddingsConfigError) as excinfo:
        embeddings.embed_texts(["Hello"], config=cfg)

    assert "Unsupported embeddings provider" in str(excinfo.value)


def test_embed_texts_openai_success_path_can_be_mocked(monkeypatch, tmp_path):
    """Happy path: OpenAI provider using a monkeypatched helper to avoid real network calls."""
    cfg = _load_minimal_config(tmp_path)

    # Configure for OpenAI.
    cfg.embeddings.provider = "openai"
    cfg.embeddings.api_key_env_var = "FIG_TEST_EMBED_KEY"

    # Ensure the env var exists so _get_api_key passes.
    monkeypatch.setenv("FIG_TEST_EMBED_KEY", "dummy-key")

    collected_args = {}

    def fake_embed_with_openai(texts, emb_cfg, api_key):
        # Capture arguments for assertions.
        collected_args["texts"] = list(texts)
        collected_args["provider"] = emb_cfg.provider
        collected_args["model"] = emb_cfg.model
        collected_args["api_key"] = api_key
        # Simple, deterministic fake embedding based on text length.
        return [[float(len(t))] for t in texts]

    # Monkeypatch the provider-specific helper so no real API call is made.
    monkeypatch.setattr(
        "fig.embeddings._embed_with_openai",
        fake_embed_with_openai,
        raising=True,
    )

    texts = ["alpha", "beta", "gamma"]
    vectors = embeddings.embed_texts(texts, config=cfg)

    # Ensure our fake helper was used with the expected arguments.
    assert collected_args["texts"] == texts
    assert collected_args["provider"] == "openai"
    assert collected_args["model"] == cfg.embeddings.model
    assert collected_args["api_key"] == "dummy-key"

    # And the result shape matches the inputs.
    assert len(vectors) == len(texts)
    assert all(isinstance(v, list) and len(v) == 1 for v in vectors)


def test_embed_texts_dummy_provider_does_not_require_api_key(tmp_path):
    """The 'dummy' provider should work without any API key or env vars."""
    cfg = _load_minimal_config(tmp_path)

    # Configure dummy provider and a non-existent env var name; it should not be used.
    cfg.embeddings.provider = "dummy"
    cfg.embeddings.api_key_env_var = "FIG_TEST_DUMMY_SHOULD_NOT_BE_USED"

    texts = ["alpha", "beta", "gamma"]
    vectors = embeddings.embed_texts(texts, config=cfg)

    assert len(vectors) == len(texts)
    for v in vectors:
        # Our dummy embedding has 4 dimensions.
        assert isinstance(v, list)
        assert len(v) == 4
        assert all(isinstance(x, float) for x in v)
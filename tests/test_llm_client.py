import copy

import pytest

from fig import llm_client


def test_generate_text_raises_config_error_when_api_key_missing(monkeypatch, config_obj):
    """If LLM is enabled but the env var is missing, raise LlmConfigError early."""
    cfg = copy.deepcopy(config_obj)
    cfg.llm.enabled = True
    cfg.llm.api_key_env_var = "FIG_TEST_MISSING_KEY"

    # Ensure the env var really isn't set.
    monkeypatch.delenv("FIG_TEST_MISSING_KEY", raising=False)

    with pytest.raises(llm_client.LlmConfigError) as excinfo:
        llm_client.generate_text("Hello", config=cfg)

    msg = str(excinfo.value)
    # Key expectations: it mentions the env var and that it's missing/empty.
    assert "FIG_TEST_MISSING_KEY" in msg
    assert "not set or empty" in msg


def test_generate_text_unsupported_provider(monkeypatch, config_obj):
    """Selecting an unsupported provider should raise a clear configuration error."""
    cfg = copy.deepcopy(config_obj)
    cfg.llm.enabled = True
    cfg.llm.provider = "unsupported-provider"

    # Ensure the configured API key env var exists so we pass the key check.
    monkeypatch.setenv(cfg.llm.api_key_env_var, "dummy-key")

    with pytest.raises(llm_client.LlmConfigError) as excinfo:
        llm_client.generate_text("Hello", config=cfg)

    assert "Unsupported LLM provider" in str(excinfo.value)
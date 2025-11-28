"""Text embeddings helper for Financial Insight Generator.

This module provides a small, provider-agnostic wrapper around text embedding
APIs. It is intentionally similar in spirit to ``fig.llm_client``:

- Configuration is read from :class:`fig.config.EmbeddingsConfig`.
- API keys are taken from environment variables (never from code or YAML).
- Errors are represented by small, explicit exception classes.
- Tests can monkeypatch the provider-specific helpers to stay fully offline.

Supported providers (via ``config.embeddings.provider``):

- ``"openai"``: uses the OpenAI embeddings API.
- ``"dummy"``: local, deterministic embedding with *no* network calls
  (useful for demos and low-resource environments).
"""

from __future__ import annotations

import os
from typing import Iterable, List

from .config import Config, EmbeddingsConfig


class EmbeddingsError(RuntimeError):
    """Base error for all embedding-related failures."""


class EmbeddingsConfigError(EmbeddingsError):
    """Raised when embeddings are disabled or misconfigured (e.g., missing API key)."""


def _get_api_key(emb_cfg: EmbeddingsConfig) -> str:
    """Fetch the API key from the environment using the configured env var name.

    This mirrors the behavior of the LLM client: configuration problems should be
    surfaced *before* making any network calls.
    """
    env_var = emb_cfg.api_key_env_var
    if not env_var:
        raise EmbeddingsConfigError(
            "Embeddings api_key_env_var is not configured; please set a valid "
            "environment variable name."
        )

    api_key = os.getenv(env_var, "")
    if not api_key:
        raise EmbeddingsConfigError(
            f"Embeddings API key environment variable '{env_var}' is not set or empty."
        )
    return api_key


def _embed_with_openai(
    texts: List[str],
    emb_cfg: EmbeddingsConfig,
    api_key: str,
) -> List[List[float]]:
    """Call the OpenAI embeddings API for the given texts.

    This function is kept small and easy to monkeypatch in tests to avoid
    hitting the real network. The actual HTTP/API interaction lives here.
    """
    # Import inside the function so that tests where we never call this path
    # don't require the dependency to be importable at collection time.
    try:
        from openai import OpenAI  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - dependency / import error path
        raise EmbeddingsConfigError(
            f"Failed to import OpenAI SDK for embeddings: {exc}"
        ) from exc

    client = OpenAI(api_key=api_key)

    try:
        response = client.embeddings.create(
            model=emb_cfg.model,
            input=texts,
        )
    except EmbeddingsError:
        # Let our own custom errors bubble up untouched.
        raise
    except Exception as exc:  # pragma: no cover - network / SDK error path
        raise EmbeddingsError(
            f"Error while calling OpenAI embeddings API: {exc}"
        ) from exc

    # The new OpenAI SDK returns an object with a ``data`` list, where each
    # item has an ``embedding`` attribute.
    embeddings: List[List[float]] = []
    for item in getattr(response, "data", []):
        vec = getattr(item, "embedding", None)
        if not isinstance(vec, Iterable):
            raise EmbeddingsError(
                "OpenAI embeddings response did not contain iterable 'embedding' vectors."
            )
        embeddings.append([float(x) for x in list(vec)])

    if len(embeddings) != len(texts):
        raise EmbeddingsError(
            "Number of embeddings returned by OpenAI does not match number of input texts."
        )

    return embeddings


def _embed_with_dummy(
    texts: List[str],
    emb_cfg: EmbeddingsConfig,  # unused, kept for future options
) -> List[List[float]]:
    """Deterministic, local-only embeddings for demos and offline usage.

    The representation is intentionally simple and fast to compute. For each
    text we build a fixed-length vector:

    [len(text), n_letters, n_digits, n_whitespace]

    This is *not* meant for semantic quality, only for demonstrating the
    vector-store pipeline end-to-end without any external services or API keys.
    """
    vectors: List[List[float]] = []
    for t in texts:
        s = t or ""
        n_letters = sum(1 for ch in s if ch.isalpha())
        n_digits = sum(1 for ch in s if ch.isdigit())
        n_spaces = sum(1 for ch in s if ch.isspace())
        vectors.append(
            [
                float(len(s)),
                float(n_letters),
                float(n_digits),
                float(n_spaces),
            ]
        )
    return vectors


def embed_texts(texts: List[str], config: Config) -> List[List[float]]:
    """Embed a list of texts using the provider configured in ``config``.

    Parameters
    ----------
    texts:
        List of input strings to embed.
    config:
        Loaded :class:`fig.config.Config` object.

    Returns
    -------
    List[List[float]]
        A list of embedding vectors, one per input text.

    Raises
    ------
    EmbeddingsConfigError
        If the provider is unsupported or required configuration (API key) is
        missing.
    EmbeddingsError
        For runtime errors from the underlying provider.
    """
    if not texts:
        return []

    emb_cfg = config.embeddings

    # Determine which provider to use.
    provider = (emb_cfg.provider or "").strip().lower() or "openai"

    # Local dummy provider: no API key, no network.
    if provider == "dummy":
        return _embed_with_dummy(texts, emb_cfg)

    # OpenAI provider (default)
    if provider == "openai":
        api_key = _get_api_key(emb_cfg)
        return _embed_with_openai(texts, emb_cfg, api_key=api_key)

    # If we reach this point, the provider is unknown / unsupported.
    raise EmbeddingsConfigError(
        f"Unsupported embeddings provider: {emb_cfg.provider!r}. "
        "Currently supported providers: 'openai', 'dummy'."
    )
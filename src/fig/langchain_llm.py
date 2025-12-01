"""LangChain-based LLM and embeddings helpers for Financial Insight Generator.

This module exposes small, focused wrappers that adapt the existing
configuration and embeddings utilities to LangChain's abstractions.

Design goals
------------
- Keep all LangChain-specific imports in one place.
- Respect the existing :class:`fig.config.LlmConfig` and EmbeddingsConfig.
- Provide:
  - `get_langchain_chat_model`  -> returns a LangChain ChatModel.
  - `FigEmbeddings` / `get_langchain_embeddings` -> LangChain `Embeddings`
    implementation that reuses :func:`fig.embeddings.embed_texts`.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

from langchain_core.embeddings import Embeddings as BaseEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from .config import Config, LlmConfig
from .embeddings import embed_texts
from .llm_client import LlmConfigError


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_enabled(llm_cfg: LlmConfig) -> None:
    """Raise LlmConfigError if LLM usage is disabled in config."""
    if not llm_cfg.enabled:
        raise LlmConfigError(
            "LLM usage is disabled in config (llm.enabled is false). "
            "Enable it in config.yaml to use LangChain-based chains."
        )


def _get_api_key(env_var: str) -> str:
    """Fetch an API key from the environment or raise LlmConfigError."""
    if not env_var:
        raise LlmConfigError(
            "llm.api_key_env_var is not configured; please set a valid "
            "environment variable name in config.yaml."
        )
    value = os.getenv(env_var)
    if not value:
        raise LlmConfigError(
            f"LLM API key environment variable {env_var!r} is not set or empty."
        )
    return value


# ---------------------------------------------------------------------------
# Chat model factory
# ---------------------------------------------------------------------------


def get_langchain_chat_model(config: Config) -> BaseChatModel:
    """Return a LangChain ChatModel configured from the given Config.

    This function mirrors :func:`fig.llm_client.generate_text` but instead of
    issuing a one-off completion, it returns a configured chat model instance
    compatible with LangChain's LCEL / Runnable pipelines.

    Currently supported providers (via ``config.llm.provider``):

    - ``"openai"``: uses :class:`langchain_openai.ChatOpenAI`.
    - ``"dummy"``: local, deterministic model for offline tests and demos.
    """
    llm_cfg = config.llm
    _ensure_enabled(llm_cfg)

    provider = (llm_cfg.provider or "").strip().lower() or "openai"

    if provider == "dummy":
        return DummyChatModel()

    if provider == "openai":
        api_key = _get_api_key(llm_cfg.api_key_env_var)
        # Note: ChatOpenAI uses the OpenAI SDK under the hood. We pass in the
        # API key explicitly rather than relying on global environment so that
        # the behavior mirrors fig.llm_client.
        return ChatOpenAI(
            model=llm_cfg.model,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            timeout=llm_cfg.timeout_seconds,
            api_key=api_key,
        )

    raise LlmConfigError(
        f"Unsupported LLM provider for LangChain engine: {llm_cfg.provider!r}. "
        "Currently supported providers are: 'openai', 'dummy'."
    )


class DummyChatModel(BaseChatModel):
    """Simple, deterministic chat model with *no* network calls.

    This is primarily intended for tests and low-resource environments where
    you still want to exercise LangChain chains without calling a real API.
    """

    @property
    def _llm_type(self) -> str:  # pragma: no cover - trivial
        return "dummy"

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a dummy response based on the last human message.

        The exact behavior is not important; it just needs to be deterministic
        and obviously fake so it does not get confused with real model output.
        """
        last_user_content: Optional[str] = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                # `content` can technically be a string or more complex type,
                # but in this project we only use simple string prompts.
                last_user_content = str(msg.content)
                break

        if last_user_content is None:
            text = "[DUMMY MODEL] No user message provided."
        else:
            text = f"[DUMMY MODEL] Echoing your request: {last_user_content}"

        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])


# ---------------------------------------------------------------------------
# Embeddings wrapper
# ---------------------------------------------------------------------------


class FigEmbeddings(BaseEmbeddings):
    """LangChain-compatible embeddings wrapper around ``fig.embeddings``.

    This class is deliberately thin: it delegates all actual embedding work to
    :func:`fig.embeddings.embed_texts`, which already knows how to read the
    correct provider configuration (OpenAI vs dummy) from :class:`Config`.

    This keeps the LangChain integration small and defers provider-specific
    API usage to the existing embeddings module.
    """

    def __init__(self, config: Config):
        self._config = config

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Delegates to :func:`fig.embeddings.embed_texts`.
        """
        return embed_texts(texts, config=self._config)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.

        Delegates to :func:`fig.embeddings.embed_texts` and returns the first
        embedding vector.
        """
        vectors = embed_texts([text], config=self._config)
        return vectors[0]


def get_langchain_embeddings(config: Config) -> FigEmbeddings:
    """Return a :class:`FigEmbeddings` instance for the given config.

    Having this small factory function keeps callers slightly more decoupled
    from the concrete embeddings implementation.
    """
    return FigEmbeddings(config=config)
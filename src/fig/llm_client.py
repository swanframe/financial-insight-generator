"""Provider-agnostic LLM client for Financial Insight Generator.

Design goals:
- Centralize all direct calls to external LLM providers in one place.
- Keep the rest of the codebase unaware of provider-specific SDK details.
- Read API keys from environment variables (never from config files or code).
- Be easy to mock in tests (simple function interface).

In later phases, other modules (llm_prompts, llm_insights, llm_chatbot) will
call `generate_text` instead of importing provider SDKs directly.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

from .config import Config, LlmConfig


class LlmError(RuntimeError):
    """Base error for all LLM-related failures."""


class LlmConfigError(LlmError):
    """Raised when LLM is disabled or misconfigured (e.g., missing API key)."""


def _get_api_key(llm_cfg: LlmConfig) -> str:
    """Fetch the API key from the environment using the configured env var name."""
    api_key = os.getenv(llm_cfg.api_key_env_var)
    if not api_key:
        raise LlmConfigError(
            f"Environment variable {llm_cfg.api_key_env_var!r} is not set or empty.\n"
            "Set this variable to your LLM provider API key (e.g., in a .env file or shell), "
            "or disable LLM features by setting llm.enabled: false in config.yaml."
        )
    return api_key


def _ensure_enabled(llm_cfg: LlmConfig) -> None:
    """Raise if the LLM features are disabled in configuration."""
    if not llm_cfg.enabled:
        raise LlmConfigError(
            "LLM features are disabled in configuration (llm.enabled is false).\n"
            "To enable them, edit config.yaml and set llm.enabled: true."
        )


def generate_text(
    user_prompt: str,
    *,
    config: Config,
    system_prompt: Optional[str] = None,
    language: Optional[str] = None,
) -> str:
    """Generate text from an LLM based on the given prompts and config.

    Parameters
    ----------
    user_prompt:
        The main user-facing prompt (instructions, question, etc.).
    config:
        The loaded Config object, which includes an LlmConfig instance.
    system_prompt:
        Optional high-level system / role message (e.g. "You are a financial analyst...").
        If provided, this will be sent as the first message in the conversation.
    language:
        Optional language hint ("en", "id", ...). It is usually better to encode
        language requirements directly in the prompt builder, but this parameter
        is available in case you want to adjust behavior per language.

    Returns
    -------
    str
        The model's textual response.

    Raises
    ------
    LlmConfigError
        If LLM is disabled, misconfigured, or the provider is not supported.
    LlmError
        If an unexpected error occurs while calling the provider SDK.
    """
    llm_cfg = config.llm

    # Ensure LLM usage is allowed by config.yaml
    _ensure_enabled(llm_cfg)
    api_key = _get_api_key(llm_cfg)

    provider = (llm_cfg.provider or "").strip().lower()
    if provider == "openai":
        return _generate_text_openai(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            llm_cfg=llm_cfg,
        )

    # In the future you can add more providers here, e.g. "gemini", "deepseek", etc.
    raise LlmConfigError(
        f"Unsupported LLM provider: {llm_cfg.provider!r}. "
        "Update fig.llm_client to add support for this provider, or switch to a supported one "
        'such as "openai" via the llm.provider setting in config.yaml.'
    )


# ---------------------------------------------------------------------------
# OpenAI provider implementation
# ---------------------------------------------------------------------------


def _generate_text_openai(
    *,
    user_prompt: str,
    system_prompt: Optional[str],
    api_key: str,
    llm_cfg: LlmConfig,
) -> str:
    """Call OpenAI's Chat Completions API using the official Python SDK.

    This implementation assumes you have installed `openai>=1.0.0` and that
    your API key is available in the environment variable configured via
    `llm.api_key_env_var` (default: OPENAI_API_KEY).

    The code path is intentionally small and focused so that you can replace
    it with another provider or a custom HTTP client if desired.
    """
    try:
        # Lazy import so that the rest of the project does not require the
        # openai package unless the OpenAI provider is actually used.
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise LlmConfigError(
            "OpenAI provider selected, but the 'openai' Python package is not installed.\n"
            "Install it with `pip install openai` or choose a different provider in config.yaml."
        ) from exc

    # Build the message list for a simple one-shot interaction.
    messages: List[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        client = OpenAI(
            api_key=api_key,
            timeout=float(llm_cfg.timeout_seconds),
        )

        completion = client.chat.completions.create(
            model=llm_cfg.model,
            messages=messages,
            temperature=float(llm_cfg.temperature),
            max_tokens=int(llm_cfg.max_tokens),
        )

        # Extract the text content from the first choice.
        choice = completion.choices[0]
        content = getattr(choice.message, "content", None)  # newer SDK style
        if isinstance(content, str):
            return content
        # Some SDK variants may return a list of parts; join them defensively.
        if isinstance(content, Iterable):
            parts = [str(part) for part in content]
            return " ".join(parts)

        # Fallback if we couldn't find a structured message content.
        return str(completion)
    except LlmError:
        # Let our own custom errors bubble up untouched.
        raise
    except Exception as exc:  # pragma: no cover - network / SDK error path
        raise LlmError(
            f"Error while calling OpenAI chat.completions.create: {exc}"
        ) from exc
"""Simple internationalization (i18n) utilities for Financial Insight Generator.

This module provides a minimal translation layer based on YAML locale files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import yaml

DEFAULT_LANGUAGE = "en"
FALLBACK_LANGUAGE = "en"

_LOCALE_CACHE: Dict[str, Dict[str, Any]] = {}


def _locales_dir() -> Path:
    """Return the directory where locale YAML files live."""
    return Path(__file__).resolve().parent / "locales"


def _load_locale(language: str) -> Dict[str, Any]:
    """Load a locale dictionary for the given language code.

    Results are cached in memory so repeated calls are cheap.
    """
    lang = (language or DEFAULT_LANGUAGE).lower()
    if lang in _LOCALE_CACHE:
        return _LOCALE_CACHE[lang]

    locales_dir = _locales_dir()
    path = locales_dir / f"{lang}.yaml"

    if not path.exists():
        data: Dict[str, Any] = {}
    else:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            data = {}

    _LOCALE_CACHE[lang] = data
    return data


def _deep_get(data: Dict[str, Any], key: str) -> Any:
    """Look up a dotted key like 'report.section.overview' in a nested dict."""
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def get_translator(language: str | None = None) -> Callable[[str], str]:
    """Return a translation function bound to the given language.

    The returned function has the signature::

        t(key: str, **kwargs) -> str

    It looks up `key` in the locale dictionaries, formatting the
    result with the provided keyword arguments. If a key is missing in
    the requested language it falls back to the fallback language, and
    finally to the key itself.
    """
    lang = (language or DEFAULT_LANGUAGE).lower()
    primary = _load_locale(lang)
    fallback = primary if lang == FALLBACK_LANGUAGE else _load_locale(FALLBACK_LANGUAGE)

    def t(key: str, **kwargs: Any) -> str:
        value = _deep_get(primary, key)
        if value is None:
            value = _deep_get(fallback, key)
        if value is None:
            # Fall back to the key so missing translations are visible
            value = key

        if not isinstance(value, str):
            return str(value)

        if kwargs:
            try:
                return value.format(**kwargs)
            except Exception:
                # If formatting fails, return the unformatted template
                return value
        return value

    return t
"""Build or update the transaction vector index for Financial Insight Generator.

This script is a small, focused entry point that:

- Loads and cleans transaction data via `fig.preprocessing.load_and_clean_transactions`.
- Reads the vector store configuration from `config.yaml`.
- Builds or updates a vector index via `fig.vector_store.build_or_update_index`.

Design goals:
- Optional and safe: if `vector_store.enabled` is false or misconfigured, the script
  prints a friendly note instead of crashing.
- i18n-aware: uses the existing `fig.i18n.get_translator` mechanism so that messages
  can be localized. If translation keys are missing, it falls back to sensible
  English defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from fig.preprocessing import load_and_clean_transactions
from fig.vector_store import (
    build_or_update_index,
    VectorStoreConfigError,
    VectorStoreError,
)
from fig.i18n import get_translator


def _t_or_default(
    t: Callable[[str], str],
    key: str,
    default: str,
    **kwargs: Any,
) -> str:
    """Translate a key, with a safe fallback to a default message.

    - First calls `t(key, **kwargs)`.
    - If the translation is missing (i.e. the result equals the key itself),
      it falls back to `default.format(**kwargs)` when possible.
    - This allows adding proper `vector.*` entries to the locale YAML files
      later, without breaking existing behaviour today.
    """
    text = t(key, **kwargs)
    if text == key:
        # Likely missing translation; fall back to the inline default.
        try:
            return default.format(**kwargs)
        except Exception:
            return default
    return text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or update the transaction vector index for "
            "Financial Insight Generator (FIG)."
        )
    )

    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Clear and rebuild the index from scratch instead of updating "
            "an existing index."
        ),
    )

    parser.add_argument(
        "--lang",
        help=(
            "Optional language override (e.g. 'en' or 'id'). "
            "Defaults to ui.language in config.yaml."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)

    # --- Run the existing data pipeline to get clean transactions + config ---
    df, pipeline_report, cfg = load_and_clean_transactions(config_path)

    # --- Determine effective language: CLI flag > config > default 'en' ---
    if hasattr(cfg, "ui"):
        config_lang = getattr(cfg.ui, "language", "en") or "en"
    else:
        config_lang = "en"

    raw_lang = args.lang or config_lang
    language = str(raw_lang).strip().lower() if raw_lang else "en"
    t = get_translator(language)

    # --- Inspect vector_store configuration ---
    vs_cfg = getattr(cfg, "vector_store", None)
    provider = getattr(vs_cfg, "provider", "chroma") if vs_cfg is not None else "chroma"
    n_documents = int(len(df))

    # If the vector store is disabled in config, print a friendly note and exit.
    if vs_cfg is None or not getattr(vs_cfg, "enabled", False):
        msg = _t_or_default(
            t,
            "vector.build_index.disabled",
            "[Vector] Vector store is disabled in config.yaml "
            "(vector_store.enabled = false).",
        )
        print(msg)
        return

    header_msg = _t_or_default(
        t,
        "vector.build_index.header",
        "[Vector] Building transaction vector index...",
    )
    print(header_msg)

    # --- Build or update the index, handling errors safely ---
    try:
        build_or_update_index(df, cfg, rebuild=args.rebuild)
        success_msg = _t_or_default(
            t,
            "vector.build_index.success",
            "[Vector] Indexed {n_documents} transactions using provider '{provider}'.",
            n_documents=n_documents,
            provider=provider,
        )
        print(success_msg)
    except VectorStoreConfigError as exc:
        error_msg = _t_or_default(
            t,
            "vector.build_index.config_error",
            "[Vector] Vector store configuration error: {error}",
            error=str(exc),
        )
        print(error_msg)
    except VectorStoreError as exc:
        error_msg = _t_or_default(
            t,
            "vector.build_index.runtime_error",
            "[Vector] Vector store runtime error: {error}",
            error=str(exc),
        )
        print(error_msg)
    except Exception as exc:
        # Catch-all to ensure the script fails gracefully instead of crashing.
        error_msg = _t_or_default(
            t,
            "vector.build_index.runtime_error",
            "[Vector] Vector store runtime error: {error}",
            error=str(exc),
        )
        print(error_msg)


if __name__ == "__main__":
    main()
"""Search the transaction vector index for similar transactions.

This script is a simple, RAG-friendly demo:

- It loads configuration from a YAML file.
- It assumes a vector index has already been built (e.g. via run_build_vector_index.py).
- It queries the vector store for transactions similar to a natural-language query.
- It prints a small, human-readable summary of the top matches.

Design goals:
- Optional and safe:
  - If vector_store.enabled is false, it prints a friendly note and exits.
  - Configuration/runtime errors from the vector store do not crash the pipeline.
- i18n-aware:
  - Uses fig.i18n.get_translator, but falls back to inline English defaults if
    translations are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, List, Optional

from fig.config import load_config
from fig.i18n import get_translator
import fig.vector_store as vector_store


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
    """
    text = t(key, **kwargs)
    if text == key:
        try:
            return default.format(**kwargs)
        except Exception:
            return default
    return text


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search the transaction vector index for similar transactions "
            "using a natural-language query."
        )
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )

    parser.add_argument(
        "-l",
        "--lang",
        type=str,
        help=(
            "Language for user-facing output (e.g., 'en' or 'id'). "
            "Defaults to ui.language in config.yaml if not provided."
        ),
    )

    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help=(
            "Natural-language search query describing the transactions to find. "
            "If omitted, you will be prompted interactively."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        help=(
            "Number of similar transactions to retrieve. "
            "Defaults to vector_store.default_top_k."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    # Determine effective language: CLI override > config > 'en'
    if hasattr(cfg, "ui"):
        config_lang = getattr(cfg.ui, "language", "en") or "en"
    else:
        config_lang = "en"
    raw_lang = args.lang or config_lang
    language = str(raw_lang).strip().lower() if raw_lang else "en"
    t = get_translator(language)

    vs_cfg = getattr(cfg, "vector_store", None)
    if vs_cfg is None or not getattr(vs_cfg, "enabled", False):
        msg = _t_or_default(
            t,
            "vector.search.disabled",
            "[Vector] Vector store is disabled in config.yaml "
            "(vector_store.enabled = false).",
        )
        print(msg)
        return

    # Determine the query text: CLI flag or interactive prompt.
    query_text = args.query
    if not query_text:
        prompt = _t_or_default(
            t,
            "vector.search.prompt",
            "Enter a search query describing the transactions to find: ",
        )
        try:
            query_text = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            msg = _t_or_default(
                t,
                "vector.search.cancelled",
                "[Vector] Search cancelled.",
            )
            print(msg)
            return

    if not query_text:
        msg = _t_or_default(
            t,
            "vector.search.no_query",
            "[Vector] No query provided; nothing to search.",
        )
        print(msg)
        return

    header = _t_or_default(
        t,
        "vector.search.header",
        "[Vector] Searching for transactions similar to: {query}",
        query=query_text,
    )
    print(header)

    # Perform the vector search.
    try:
        matches = vector_store.query_similar_transactions(
            query_text,
            config=cfg,
            top_k=args.top_k,
        )
    except vector_store.VectorStoreConfigError as exc:
        msg = _t_or_default(
            t,
            "vector.search.config_error",
            "[Vector] Vector store configuration error: {error}",
            error=str(exc),
        )
        print(msg)
        return
    except vector_store.VectorStoreError as exc:
        msg = _t_or_default(
            t,
            "vector.search.runtime_error",
            "[Vector] Vector store runtime error: {error}",
            error=str(exc),
        )
        print(msg)
        return
    except Exception as exc:
        msg = _t_or_default(
            t,
            "vector.search.runtime_error",
            "[Vector] Vector store runtime error: {error}",
            error=str(exc),
        )
        print(msg)
        return

    if not matches:
        msg = _t_or_default(
            t,
            "vector.search.no_results",
            "[Vector] No matching transactions found. The index may be empty or "
            "not yet built.",
        )
        print(msg)
        return

    results_header = _t_or_default(
        t,
        "vector.search.results_header",
        "[Vector] Top {n} similar transactions:",
        n=len(matches),
    )
    print(results_header)

    for idx, match in enumerate(matches, start=1):
        meta = match.metadata or {}
        date = meta.get("date") or "?"
        category = meta.get("category") or "?"
        product = meta.get("product") or "?"
        amount = meta.get("amount")
        amount_str = f"{amount}" if amount is not None else "?"
        txn_id = meta.get("transaction_id") or match.id
        score_str = f"{match.score:.3f}"

        line = _t_or_default(
            t,
            "vector.search.result_line",
            "{idx}. {date} | id={transaction_id} | "
            "category={category} | product={product} | "
            "amount={amount} | similarity={score}",
            idx=idx,
            date=date,
            transaction_id=txn_id,
            category=category,
            product=product,
            amount=amount_str,
            score=score_str,
        )
        print(line)


if __name__ == "__main__":
    main()
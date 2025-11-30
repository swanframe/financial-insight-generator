from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logging
import os

import pandas as pd

from .config import Config
from .retrieval_schema import RetrievalContext, build_retrieval_context
from .vector_store import (
    TransactionMatch,
    VectorStoreConfigError,
    VectorStoreError,
    build_or_update_index,
    query_similar_transactions,
)

logger = logging.getLogger(__name__)


def _rag_debug_enabled() -> bool:
    """Return True if verbose RAG debugging is enabled via FIG_DEBUG_RAG.

    This helper is intentionally simple and local to the retrieval module so
    that developers can opt into extra visibility without changing config.yaml:

        export FIG_DEBUG_RAG=1
    """
    value = os.environ.get("FIG_DEBUG_RAG", "")
    if not value:
        return False
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class RetrievalStatus:
    """Lightweight status object describing the outcome of an index build.

    This is intentionally simple: it is mainly useful for debugging and for
    future CLI hooks. For now, most callers can ignore it and just rely on
    exceptions or return values.

    Attributes
    ----------
    attempted:
        Whether we actually attempted to build/update the index. For example,
        if the vector store is disabled and ``fail_silently=True``, this will
        be False.
    success:
        True if the build completed without errors. False if an exception was
        caught and suppressed because ``fail_silently=True``.
    message:
        Optional human-readable message, suitable for logging or debug prints.
    """

    attempted: bool
    success: bool
    message: Optional[str] = None


def is_retrieval_enabled(config: Config) -> bool:
    """Return True if the vector store is enabled in config.

    This does *not* verify API keys or connectivity; it simply reflects the
    boolean ``vector_store.enabled`` flag. More detailed configuration errors
    (e.g. missing embeddings API key) will be surfaced when actually building
    or querying the index.
    """
    return bool(getattr(config, "vector_store", None) and config.vector_store.enabled)


def build_transaction_index_from_dataframe(
    df: pd.DataFrame,
    *,
    config: Config,
    rebuild: bool = False,
    fail_silently: bool = True,
) -> RetrievalStatus:
    """Build or update the transaction vector index from a cleaned DataFrame.

    This is a thin wrapper around :func:`fig.vector_store.build_or_update_index`
    that adds a small amount of convenience and error-handling.

    Parameters
    ----------
    df:
        Cleaned transactions DataFrame using the internal column names
        (``date``, ``amount``, ``category``, ``product``, ``customer_id``,
        ``channel``, etc.). This should typically be the same DataFrame used to
        compute analytics and metrics.
    config:
        The loaded :class:`fig.config.Config` object.
    rebuild:
        If True, clear any existing vectors before inserting new ones.
    fail_silently:
        If True (default), configuration/runtime errors from the vector store
        will be caught and reflected in the returned :class:`RetrievalStatus`
        instead of being raised. If False, errors are propagated.

    Returns
    -------
    RetrievalStatus
        A simple status summary. Callers that only care about success/failure
        can check the ``success`` flag.
    """
    provider = getattr(getattr(config, "vector_store", None), "provider", None)
    n_rows = getattr(df, "shape", (None,))[0]

    logger.debug(
        "FIG RAG: starting transaction index build "
        "(enabled=%s, provider=%r, rebuild=%s, rows=%s)",
        is_retrieval_enabled(config),
        provider,
        rebuild,
        n_rows,
    )

    if not is_retrieval_enabled(config):
        msg = (
            "Vector store is disabled (vector_store.enabled = false); "
            "skipping index build."
        )
        logger.debug("FIG RAG: %s", msg)
        if fail_silently:
            if _rag_debug_enabled():
                print(f"[FIG RAG] {msg}")
            return RetrievalStatus(attempted=False, success=False, message=msg)
        raise VectorStoreConfigError(msg)

    try:
        build_or_update_index(df=df, config=config, rebuild=rebuild)
    except VectorStoreConfigError as exc:
        logger.warning("FIG RAG: index build failed due to configuration: %s", exc)
        if fail_silently:
            msg = str(exc)
            if _rag_debug_enabled():
                print(f"[FIG RAG] Vector index build failed: {msg}")
            return RetrievalStatus(
                attempted=True,
                success=False,
                message=msg,
            )
        raise
    except VectorStoreError as exc:
        logger.warning("FIG RAG: index build failed due to runtime error: %s", exc)
        if fail_silently:
            msg = str(exc)
            if _rag_debug_enabled():
                print(f"[FIG RAG] Vector index build failed: {msg}")
            return RetrievalStatus(
                attempted=True,
                success=False,
                message=msg,
            )
        raise

    logger.info(
        "FIG RAG: transaction index build completed (provider=%r, rows=%s, rebuild=%s)",
        provider,
        n_rows,
        rebuild,
    )
    if _rag_debug_enabled():
        print(
            f"[FIG RAG] Built vector index for {n_rows} rows using "
            f"provider '{provider}' (rebuild={rebuild})"
        )

    return RetrievalStatus(attempted=True, success=True, message=None)


def retrieve_transactions_for_query(
    query_text: str,
    *,
    config: Config,
    top_k: Optional[int] = None,
    language: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    fail_silently: bool = True,
) -> Optional[RetrievalContext]:
    """Run a similarity search for transactions and return a RetrievalContext.

    This function is the main entry point for RAG-style features in FIG. It:

    - Checks whether the vector store is enabled.
    - Delegates to :func:`fig.vector_store.query_similar_transactions`.
    - Wraps the resulting :class:`TransactionMatch` objects into a
      :class:`RetrievalContext` for easy consumption by LLM prompts.

    Parameters
    ----------
    query_text:
        Natural-language query (e.g. ``"large transactions in March"``).
    config:
        Loaded :class:`fig.config.Config` object.
    top_k:
        Optional override for the number of neighbours to retrieve. When None,
        the vector_store.default_top_k setting is used.
    language:
        Optional ISO language code for the *user-facing* language ("en", "id",
        ...). The underlying index is still English, but prompts may want this
        to decide how to render summaries.
    filters:
        Optional high-level filters (e.g. month, category) for logging or
        prompt context. This function does not currently apply these filters
        itself; they are stored on the returned :class:`RetrievalContext` for
        downstream use.
    fail_silently:
        If True (default), configuration/runtime errors from the vector store
        will result in ``None`` being returned. This is convenient for callers
        that want to "try RAG, otherwise fall back". If False, such errors are
        propagated.

    Returns
    -------
    RetrievalContext or None
        - A :class:`RetrievalContext` when retrieval runs successfully.
        - ``None`` if the vector store is disabled or an error occurs and
          ``fail_silently=True`` is set.
    """
    provider = getattr(getattr(config, "vector_store", None), "provider", None)
    logger.debug(
        "FIG RAG: starting transaction retrieval (query=%r, top_k=%s, provider=%r)",
        query_text,
        top_k,
        provider,
    )

    if not is_retrieval_enabled(config):
        msg = (
            "Vector store is disabled (vector_store.enabled = false); "
            "cannot run transaction retrieval."
        )
        logger.debug("FIG RAG: %s", msg)
        if fail_silently:
            if _rag_debug_enabled():
                print(f"[FIG RAG] {msg}")
            return None
        raise VectorStoreConfigError(msg)

    try:
        matches: List[TransactionMatch] = query_similar_transactions(
            query_text=query_text,
            config=config,
            top_k=top_k,
        )
    except (VectorStoreConfigError, VectorStoreError) as exc:
        logger.warning(
            "FIG RAG: transaction retrieval failed (query=%r): %s",
            query_text,
            exc,
        )
        if fail_silently:
            if _rag_debug_enabled():
                print(f"[FIG RAG] Transaction retrieval failed: {exc}")
            # Callers can log this message if they need more detail.
            return None
        raise

    logger.debug(
        "FIG RAG: transaction retrieval returned %d matches for query %r",
        len(matches),
        query_text,
    )
    if _rag_debug_enabled():
        preview = ", ".join(
            f"{m.id}:{m.score:.3f}" for m in matches[:3]
        )
        print(
            f"[FIG RAG] Retrieved {len(matches)} matches for query {query_text!r} "
            f"(top3: {preview})"
        )

    # Even if there are no matches, we still return a RetrievalContext; this
    # lets callers distinguish "RAG ran but found nothing" from "RAG disabled".
    return build_retrieval_context(
        matches=matches,
        query=query_text,
        language=language,
        top_k=top_k,
        filters=filters or {},
    )
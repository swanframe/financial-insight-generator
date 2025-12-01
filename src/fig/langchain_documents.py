"""LangChain document helpers for Financial Insight Generator.

This module provides small, reusable helpers to map FIG's internal data
structures into LangChain :class:`~langchain_core.documents.Document` objects.

There are two main entry points:

- ``dataframe_to_documents`` converts the cleaned transaction DataFrame into
  Documents using the same human-readable text and metadata that the vector
  store uses for indexing.
- ``retrieval_context_to_documents`` converts a high-level
  :class:`fig.retrieval_schema.RetrievalContext` (used throughout the existing
  RAG pipeline) into Documents suitable for LangChain retrievers and chains.

The goal is to keep all of the "how do we describe a transaction?" decisions
in one place so that both the native pipeline and the LangChain integration
stay in sync.
"""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd
from langchain_core.documents import Document

from .config import Config
from .retrieval_schema import RetrievalContext, RetrievedTransaction
from .vector_store import TransactionDocument, _build_transaction_documents


def dataframe_to_documents(df: pd.DataFrame, config: Optional[Config] = None) -> List[Document]:
    """Convert a cleaned transactions DataFrame into LangChain Documents.

    Parameters
    ----------
    df:
        Cleaned transactions DataFrame, as produced by :mod:`fig.preprocessing`.
        Expected to contain the internal columns:

        - ``date``
        - ``amount``
        - optional: ``cost``, ``category``, ``product``, ``customer_id``,
          ``channel``

    config:
        Currently unused but accepted for future extensions (e.g. language-
        specific formatting or configurable metadata enrichment). It is kept
        here so the signature matches other helpers where ``Config`` is the
        primary dependency injection mechanism.

    Returns
    -------
    List[Document]
        One :class:`Document` per row in the DataFrame.
    """
    # Reuse the existing vector store logic to avoid duplicating the way we
    # describe transactions. This ensures that:
    #
    # - The text used for semantic search embeddings matches the text we show
    #   as `page_content` in Documents.
    # - The metadata keys stay consistent across the native and LangChain
    #   pipelines.
    tx_docs: List[TransactionDocument] = _build_transaction_documents(df)

    documents: List[Document] = []
    for tx in tx_docs:
        # `tx.metadata` already contains:
        # - transaction_id
        # - date
        # - amount
        # - category
        # - product
        # - customer_id
        # - channel
        # - row_index
        documents.append(
            Document(
                page_content=tx.text,
                metadata=dict(tx.metadata),
            )
        )

    return documents


def _format_retrieved_transaction_text(rt: RetrievedTransaction) -> str:
    """Build a concise, English summary for a retrieved transaction.

    This is only used as a fallback when the original free-form `text` is
    not available on the :class:`RetrievedTransaction` object.
    """
    parts: List[str] = []

    if rt.date is not None:
        parts.append(f"date={rt.date}")
    if rt.amount is not None:
        parts.append(f"amount={rt.amount}")
    if rt.category:
        parts.append(f"category={rt.category}")
    if rt.product:
        parts.append(f"product={rt.product}")
    if rt.customer_id:
        parts.append(f"customer_id={rt.customer_id}")
    if rt.channel:
        parts.append(f"channel={rt.channel}")

    if not parts:
        return f"Retrieved transaction {rt.transaction_id or rt.id}"

    return f"Retrieved transaction {rt.transaction_id or rt.id}: " + ", ".join(parts)


def retrieval_context_to_documents(ctx: RetrievalContext) -> List[Document]:
    """Convert a :class:`RetrievalContext` into LangChain Documents.

    Each match in ``ctx.matches`` becomes a single :class:`Document` with:

    - ``page_content``: either the original free-form ``text`` (if present)
      or a concise, machine-generated summary built from the structured
      fields.
    - ``metadata``: a superset of the structured and low-level fields, such as
      transaction identifiers, similarity score, query, and any extra
      metadata coming from the vector store.
    """
    documents: List[Document] = []

    for rt in ctx.matches:
        text = rt.text or _format_retrieved_transaction_text(rt)

        # Start with the promoted structured fields.
        metadata: dict[str, Any] = {
            "id": rt.id,
            "transaction_id": rt.transaction_id or rt.id,
            "score": rt.score,
            "date": rt.date,
            "amount": rt.amount,
            "category": rt.category,
            "product": rt.product,
            "customer_id": rt.customer_id,
            "channel": rt.channel,
            "row_index": rt.row_index,
        }

        # Enrich with context-level information.
        metadata["query"] = ctx.query
        metadata["language"] = ctx.language
        metadata["top_k"] = ctx.effective_top_k()
        metadata["filters"] = dict(ctx.filters)

        # Merge in the original metadata from the vector store, without
        # overwriting the promoted fields.
        for key, value in rt.metadata.items():
            metadata.setdefault(key, value)

        documents.append(Document(page_content=text, metadata=metadata))

    return documents
"""LangChain retriever built on top of FIG's existing RAG layer.

This module provides a :class:`TransactionsRetriever` that adapts the
high-level :func:`fig.retrieval.retrieve_transactions_for_query` helper to
LangChain's :class:`~langchain_core.retrievers.BaseRetriever` interface.

Key design goals
----------------
- Reuse the existing vector store and retrieval logic (including Chroma).
- Avoid duplicating any "how do we build queries / metadata?" decisions.
- Provide a clean, LangChain-friendly API for use in LCEL / Runnable chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .config import Config as FigConfig
from .langchain_documents import retrieval_context_to_documents
from .retrieval import retrieve_transactions_for_query


@dataclass
class TransactionsRetrieverConfig:
    """Lightweight configuration for :class:`TransactionsRetriever`.

    Parameters
    ----------
    language:
        Language code to use when retrieving and formatting context. This
        should match ``config.ui.language`` in most cases (e.g. ``"en"`` or
        ``"id"``).

    top_k:
        Default number of similar transactions to retrieve. If ``None``,
        the value from ``config.vector_store.default_top_k`` is used.

    fail_silently:
        When ``True`` (the default), retrieval failures or a disabled vector
        store result in an empty list of documents rather than an exception.

    filters:
        Optional filter dictionary passed through to
        :func:`fig.retrieval.retrieve_transactions_for_query`.
    """

    language: str = "en"
    top_k: Optional[int] = None
    fail_silently: bool = True
    filters: Dict[str, Any] | None = None


class TransactionsRetriever(BaseRetriever):
    """LangChain retriever over FIG's transaction vector index.

    This retriever delegates all heavy lifting to
    :func:`fig.retrieval.retrieve_transactions_for_query` and converts the
    resulting :class:`fig.retrieval_schema.RetrievalContext` into
    :class:`~langchain_core.documents.Document` objects.

    The underlying vector store can be either:

    - In-memory (for tests / demos), or
    - Chroma (persistent on disk),

    depending on :class:`fig.config.VectorStoreConfig`.

    Note: This class is a Pydantic model via ``BaseRetriever``. We therefore
    avoid overriding ``__init__`` and instead declare fields directly using
    Pydantic v2's ``model_config`` for settings.
    """

    # Pydantic model fields
    config: FigConfig
    retriever_config: TransactionsRetrieverConfig

    # Pydantic v2 configuration: allow arbitrary (non-Pydantic) types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def language(self) -> str:
        """Language used for retrieval and formatting."""
        return self.retriever_config.language

    @language.setter
    def language(self, value: str) -> None:
        self.retriever_config.language = value

    @property
    def top_k(self) -> Optional[int]:
        """Default number of neighbours to retrieve."""
        return self.retriever_config.top_k

    @top_k.setter
    def top_k(self, value: Optional[int]) -> None:
        self.retriever_config.top_k = value

    # BaseRetriever API -------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None,  # pragma: no cover - signature only
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve Documents relevant to the given query string.

        Extra keyword arguments may override ``top_k`` and ``filters`` on a
        per-call basis:

        Parameters
        ----------
        query:
            Free-text query string describing what the user is looking for.

        top_k (kwarg, optional):
            If provided, overrides the default ``top_k`` configured on this
            retriever instance.

        filters (kwarg, optional):
            Additional filters merged on top of the instance-level filters.
        """
        cfg = self.retriever_config

        # Per-call overrides for top_k and filters.
        top_k_override = kwargs.get("top_k")
        filters_override = kwargs.get("filters") or {}

        if top_k_override is not None:
            try:
                top_k_value = int(top_k_override)
            except (TypeError, ValueError):
                top_k_value = cfg.top_k
        else:
            top_k_value = cfg.top_k

        # Merge filters: instance-level filters take precedence unless the
        # per-call filters explicitly override a key.
        base_filters = cfg.filters or {}
        merged_filters: Dict[str, Any] = dict(base_filters)
        for key, value in dict(filters_override).items():
            merged_filters[key] = value

        ctx = retrieve_transactions_for_query(
            query_text=query,
            config=self.config,
            language=cfg.language,
            top_k=top_k_value,
            filters=merged_filters,
            fail_silently=cfg.fail_silently,
        )

        if ctx is None:
            # When fail_silently=True, retrieval will already have logged any
            # issues (e.g. vector store disabled, index missing) and returned
            # None. LangChain Retrievers are expected to return an empty list
            # in that case.
            return []

        return retrieval_context_to_documents(ctx)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def get_transactions_retriever(
    config: FigConfig,
    language: Optional[str] = None,
    top_k: Optional[int] = None,
    fail_silently: bool = True,
    filters: Optional[Dict[str, Any]] = None,
) -> TransactionsRetriever:
    """Create a :class:`TransactionsRetriever` for the given configuration.

    Parameters
    ----------
    config:
        Global FIG configuration object.

    language:
        Optional override for the language used by the retriever. When not
        provided, ``config.ui.language`` is used.

    top_k:
        Optional override for the default number of neighbours to retrieve.
        When ``None``, ``config.vector_store.default_top_k`` is used.

    fail_silently:
        When ``True`` (the default), retrieval failures or a disabled vector
        store result in an empty list of documents rather than raising.

    filters:
        Optional initial filter dictionary.

    Returns
    -------
    TransactionsRetriever
        A ready-to-use retriever instance suitable for LangChain chains.
    """
    effective_language = language or config.ui.language
    effective_top_k = (
        top_k if top_k is not None else config.vector_store.default_top_k
    )

    retr_cfg = TransactionsRetrieverConfig(
        language=effective_language,
        top_k=effective_top_k,
        fail_silently=fail_silently,
        filters=filters or {},
    )

    return TransactionsRetriever(config=config, retriever_config=retr_cfg)
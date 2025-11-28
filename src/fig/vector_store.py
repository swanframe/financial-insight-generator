"""Vector store integration for Financial Insight Generator.

This module provides a small, provider-agnostic abstraction over vector
databases. It is designed to:

- Work with the cleaned transaction DataFrame (internal column names).
- Use the shared embeddings helper (fig.embeddings.embed_texts).
- Support both a persistent Chroma-backed store and an in-memory implementation.
- Be safe and optional: if vector_store.enabled is False, callers get a clear
  configuration error instead of a crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import Config
from .embeddings import EmbeddingsConfigError, EmbeddingsError, embed_texts


class VectorStoreError(RuntimeError):
    """Base error for all vector-store-related failures."""


class VectorStoreConfigError(VectorStoreError):
    """Raised when vector store is disabled or misconfigured."""


@dataclass
class TransactionDocument:
    """Representation of a single transaction as an indexable document."""

    id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class TransactionMatch:
    """A similarity search result for a transaction."""

    id: str
    score: float
    metadata: Dict[str, Any]
    text: str | None = None


@dataclass
class _InMemoryStore:
    """Simple in-process store used for tests and lightweight demos."""

    ids: List[str] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)
    metadatas: List[Dict[str, Any]] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)

    def clear(self) -> None:
        self.ids.clear()
        self.embeddings.clear()
        self.metadatas.clear()
        self.documents.clear()

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ) -> None:
        """Upsert items into the in-memory store."""
        index_by_id = {id_: i for i, id_ in enumerate(self.ids)}
        for id_, emb, meta, doc in zip(ids, embeddings, metadatas, documents):
            if id_ in index_by_id:
                idx = index_by_id[id_]
                self.embeddings[idx] = emb
                self.metadatas[idx] = meta
                self.documents[idx] = doc
            else:
                self.ids.append(id_)
                self.embeddings.append(emb)
                self.metadatas.append(meta)
                self.documents.append(doc)

    def query(self, query_embedding: List[float], top_k: int) -> List[TransactionMatch]:
        """Return the top-k matches based on cosine similarity."""
        if not self.ids or not self.embeddings:
            return []

        scores: List[Tuple[float, int]] = []
        for idx, emb in enumerate(self.embeddings):
            score = _cosine_similarity(query_embedding, emb)
            scores.append((score, idx))

        # Sort by score descending (higher cosine similarity is better).
        scores.sort(key=lambda x: x[0], reverse=True)
        top_scores = scores[: max(0, int(top_k))]

        results: List[TransactionMatch] = []
        for score, idx in top_scores:
            results.append(
                TransactionMatch(
                    id=self.ids[idx],
                    score=score,
                    metadata=self.metadatas[idx],
                    text=self.documents[idx],
                )
            )
        return results


# Global registry for in-memory stores keyed by (persist_path, collection_name).
_IN_MEMORY_STORES: Dict[Tuple[str, str], _InMemoryStore] = {}


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector is all zeros or empty.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (sqrt(norm_a) * sqrt(norm_b))


def _ensure_enabled(config: Config) -> None:
    """Raise a configuration error if the vector store is disabled."""
    if not config.vector_store.enabled:
        raise VectorStoreConfigError(
            "Vector store is disabled in config.yaml (vector_store.enabled = false)."
        )


def _build_transaction_documents(df: pd.DataFrame) -> List[TransactionDocument]:
    """Convert the cleaned transaction DataFrame into indexable documents.

    Expects the internal column names produced by fig.preprocessing:
    - 'date'
    - 'amount'
    - optional: 'cost', 'category', 'product', 'customer_id', 'channel'
    """
    docs: List[TransactionDocument] = []

    for idx, row in df.iterrows():
        txn_id = f"txn-{idx}"

        date_val = row.get("date")
        if hasattr(date_val, "isoformat"):
            date_str = date_val.isoformat()
        else:
            date_str = str(date_val) if date_val is not None else ""

        amount_val = row.get("amount")
        try:
            amount_float = float(amount_val) if amount_val is not None else None
        except Exception:
            amount_float = None

        category = row.get("category")
        product = row.get("product")
        customer_id = row.get("customer_id")
        channel = row.get("channel")

        # Build a short, human-readable description for semantic search.
        parts: List[str] = []
        if date_str:
            parts.append(f"on {date_str}")
        if category:
            parts.append(f"in category {category}")
        if product:
            parts.append(f"for product {product}")
        if amount_float is not None:
            parts.append(f"with amount {amount_float}")
        if customer_id:
            parts.append(f"for customer {customer_id}")
        if channel:
            parts.append(f"via channel {channel}")

        if parts:
            text = f"Transaction {txn_id} " + ", ".join(parts) + "."
        else:
            text = f"Transaction {txn_id}."

        metadata: Dict[str, Any] = {
            "transaction_id": txn_id,
            "date": date_str,
            "amount": amount_float,
            "category": category,
            "product": product,
            "customer_id": customer_id,
            "channel": channel,
            "row_index": int(idx) if isinstance(idx, (int, float)) else idx,
        }

        docs.append(TransactionDocument(id=txn_id, text=text, metadata=metadata))

    return docs


def _get_in_memory_store(config: Config) -> _InMemoryStore:
    """Return the in-memory store instance for the given config."""
    vs_cfg = config.vector_store
    key = (str(vs_cfg.persist_path), vs_cfg.collection_name)
    store = _IN_MEMORY_STORES.get(key)
    if store is None:
        store = _InMemoryStore()
        _IN_MEMORY_STORES[key] = store
    return store


def _get_chroma_collection(config: Config):
    """Return a Chroma collection object for the given config.

    The chromadb dependency is imported lazily so that environments which do not
    use Chroma (e.g. tests using the in-memory provider) do not need it.
    """
    try:
        import chromadb  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - import error path
        raise VectorStoreConfigError(
            f"Failed to import 'chromadb' package: {exc}. "
            "Install it with `pip install chromadb` or choose provider: 'in_memory'."
        ) from exc

    vs_cfg = config.vector_store

    try:
        client = chromadb.PersistentClient(path=str(vs_cfg.persist_path))
        collection = client.get_or_create_collection(name=vs_cfg.collection_name)
    except Exception as exc:  # pragma: no cover - provider runtime issues
        raise VectorStoreError(
            f"Failed to open or create Chroma collection {vs_cfg.collection_name!r}: {exc}"
        ) from exc

    return client, collection


def build_or_update_index(
    df: pd.DataFrame,
    config: Config,
    rebuild: bool = False,
) -> None:
    """Build or update the transaction vector index based on the given DataFrame.

    Parameters
    ----------
    df:
        Cleaned transaction DataFrame with internal column names.
    config:
        Loaded configuration object.
    rebuild:
        If True, the underlying collection/index is cleared before upserting
        the new vectors.
    """
    _ensure_enabled(config)

    vs_cfg = config.vector_store
    provider = (vs_cfg.provider or "").strip().lower() or "chroma"

    docs = _build_transaction_documents(df)
    if not docs:
        # Nothing to index; this is not an error.
        return

    # Generate embeddings for all documents via the shared embedding helper.
    try:
        vectors = embed_texts([doc.text for doc in docs], config=config)
    except EmbeddingsConfigError as exc:
        # Configuration problems with the embedding provider are surfaced as
        # vector-store configuration issues.
        raise VectorStoreConfigError(
            f"Embeddings are misconfigured for vector store usage: {exc}"
        ) from exc
    except EmbeddingsError as exc:
        raise VectorStoreError(f"Error while generating embeddings for index: {exc}") from exc

    if len(vectors) != len(docs):
        raise VectorStoreError(
            "embed_texts returned an unexpected number of vectors for index build."
        )

    ids = [doc.id for doc in docs]
    texts = [doc.text for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    if provider == "in_memory":
        store = _get_in_memory_store(config)
        if rebuild:
            store.clear()
        store.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)
        return

    if provider == "chroma":
        client, collection = _get_chroma_collection(config)

        if rebuild:
            try:
                # Delete all existing items in the collection.
                collection.delete(where={})  # type: ignore[attr-defined]
            except Exception:
                # If delete fails (e.g., older Chroma version), ignore and proceed
                # with upsert, which will overwrite existing IDs where possible.
                pass

        try:
            collection.upsert(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=texts,
            )
            # In scripts, Chroma persists automatically when the client is
            # destroyed, but we call persist() for extra safety in notebooks.
            try:
                client.persist()  # type: ignore[attr-defined]
            except Exception:
                # Not all client implementations require or support persist().
                pass
        except Exception as exc:  # pragma: no cover - provider runtime issues
            raise VectorStoreError(f"Failed to upsert vectors into Chroma: {exc}") from exc
        return

    # Unknown provider
    raise VectorStoreConfigError(
        f"Unsupported vector_store provider: {vs_cfg.provider!r}. "
        "Supported providers: 'chroma', 'in_memory'."
    )


def query_similar_transactions(
    query_text: str,
    config: Config,
    top_k: int | None = None,
) -> List[TransactionMatch]:
    """Query the vector store for transactions similar to the given text.

    Parameters
    ----------
    query_text:
        Natural-language query describing the kind of transactions you're
        looking for (e.g., "large electronics orders in March").
    config:
        Loaded configuration object.
    top_k:
        Optional override for the number of neighbours to retrieve. If None,
        ``config.vector_store.default_top_k`` is used.

    Returns
    -------
    List[TransactionMatch]
        Ranked list of similar transactions. The list may be empty if the
        index is empty or no matches are found.
    """
    _ensure_enabled(config)

    vs_cfg = config.vector_store
    provider = (vs_cfg.provider or "").strip().lower() or "chroma"
    n_results = int(top_k) if top_k is not None else int(vs_cfg.default_top_k)

    if n_results <= 0:
        return []

    # Compute the query embedding using the same embeddings configuration.
    try:
        [query_embedding] = embed_texts([query_text], config=config)
    except EmbeddingsConfigError as exc:
        raise VectorStoreConfigError(
            f"Embeddings are misconfigured for vector store usage: {exc}"
        ) from exc
    except EmbeddingsError as exc:
        raise VectorStoreError(f"Error while generating embedding for query: {exc}") from exc

    if provider == "in_memory":
        store = _get_in_memory_store(config)
        return store.query(query_embedding=query_embedding, top_k=n_results)

    if provider == "chroma":
        _, collection = _get_chroma_collection(config)

        try:
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:  # pragma: no cover - provider runtime issues
            raise VectorStoreError(f"Chroma query failed: {exc}") from exc

        ids_list = result.get("ids") or [[]]
        docs_list = result.get("documents") or [[]]
        metas_list = result.get("metadatas") or [[]]
        distances_list = result.get("distances") or [[]]

        ids = ids_list[0] if ids_list else []
        docs = docs_list[0] if docs_list else []
        metas = metas_list[0] if metas_list else []
        distances = distances_list[0] if distances_list else []

        matches: List[TransactionMatch] = []
        for id_, doc, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns distances, where smaller is better. Convert to a
            # simple "score" in [0, 1] range for human-friendly display.
            try:
                distance = float(dist)
            except Exception:
                distance = 0.0
            score = 1.0 / (1.0 + max(distance, 0.0))
            matches.append(
                TransactionMatch(
                    id=str(id_),
                    score=score,
                    metadata=dict(meta or {}),
                    text=str(doc) if doc is not None else None,
                )
            )
        return matches

    # Unknown provider
    raise VectorStoreConfigError(
        f"Unsupported vector_store provider: {vs_cfg.provider!r}. "
        "Supported providers: 'chroma', 'in_memory'."
    )
"""RAG retrieval schema and helper types for Financial Insight Generator.

This module defines small, well-documented data structures that sit on top of
the lower-level vector_store API:

- TransactionMatch (from fig.vector_store) represents a raw similarity result.
- RetrievedTransaction captures the fields we actually want to expose to prompts.
- RetrievalContext is the high-level bundle that report/chat prompts will see.

Phase 1 is intentionally *schema-only*:
- No actual calls to embed_texts or query_similar_transactions are made here.
- Higher-level retrieval helpers will be added in later phases, reusing these
  types so that prompts and tests have a stable contract to rely on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fig.vector_store import TransactionMatch


@dataclass
class RetrievedTransaction:
    """Structured view of a single transaction returned by vector search.

    This is a lightly processed wrapper around `TransactionMatch` that makes it
    easier to use RAG results inside prompts:

    - Common fields (date, amount, category, etc.) are promoted from `metadata`
      to first-class attributes.
    - The original `metadata` dict is preserved for extensibility.
    - The original free-form `text` (if present) is kept for reference, but
      prompt code is encouraged to build its own, language-aware summaries.

    The metadata keys below intentionally mirror what `build_or_update_index`
    currently stores in `fig.vector_store`:

    - "transaction_id": str
    - "date": str (ISO-like string)
    - "amount": float
    - "category": Optional[str]
    - "product": Optional[str]
    - "customer_id": Optional[str]
    - "channel": Optional[str]
    - "row_index": Optional[int | str]
    """

    # Core identifiers / scoring
    id: str
    score: float

    # Common metadata fields (promoted from metadata)
    transaction_id: Optional[str] = None
    date: Optional[str] = None
    amount: Optional[float] = None
    category: Optional[str] = None
    product: Optional[str] = None
    customer_id: Optional[str] = None
    channel: Optional[str] = None
    row_index: Optional[object] = None  # index in the cleaned DataFrame, if known

    # Original free-form document text (English description used for embeddings)
    text: Optional[str] = None

    # Full metadata, for backward compatibility / future extensions
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalContext:
    """High-level RAG context passed into LLM prompts.

    This is the main object that future RAG-enabled prompt builders will see.
    It is intentionally simple and JSON-serialisable.

    Attributes
    ----------
    query:
        The natural-language query that produced these results, if any.
        For report-style RAG (no direct user question), this may be None or a
        synthetic description of why the retrieval was run.
    matches:
        A list of `RetrievedTransaction` results, in descending similarity
        order (best match first).
    top_k:
        The requested number of matches. Defaults to len(matches) if omitted.
    language:
        Optional ISO language code ("en", "id", ...) indicating the user-facing
        language. Prompts can use this to decide how to render summaries, but
        the underlying index text is still English.
    filters:
        Optional high-level filters (e.g. month, category) that were applied
        before vector search. Stored as a plain dictionary so it can be logged,
        serialized, or shown in debug output.
    """

    query: Optional[str]
    matches: List[RetrievedTransaction]

    top_k: Optional[int] = None
    language: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)

    def effective_top_k(self) -> int:
        """Return the effective top_k for this context.

        Falls back to len(matches) if top_k was not explicitly provided.
        """
        return self.top_k if self.top_k is not None else len(self.matches)

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Convert this context into a plain dict suitable for JSON / templating.

        Prompts are free to ignore this and build custom text, but when we want
        to show RAG evidence in a structured way (e.g. in a system/user prompt),
        this provides a stable, machine-friendly shape.
        """
        return {
            "query": self.query,
            "top_k": self.effective_top_k(),
            "language": self.language,
            "filters": dict(self.filters),
            "matches": [
                {
                    "id": m.id,
                    "score": m.score,
                    "transaction_id": m.transaction_id or m.id,
                    "date": m.date,
                    "amount": m.amount,
                    "category": m.category,
                    "product": m.product,
                    "customer_id": m.customer_id,
                    "channel": m.channel,
                    "row_index": m.row_index,
                }
                for m in self.matches
            ],
        }


def build_retrieval_context(
    matches: List[TransactionMatch],
    *,
    query: Optional[str] = None,
    language: Optional[str] = None,
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> RetrievalContext:
    """Create a `RetrievalContext` from low-level `TransactionMatch` objects.

    This helper is intentionally lightweight and does *not* perform any vector
    search itself. It simply normalises the metadata coming from the vector
    store into `RetrievedTransaction` instances.

    Later phases will:

    - Call `fig.vector_store.query_similar_transactions(...)` to obtain
      `TransactionMatch` objects.
    - Pipe those into `build_retrieval_context(...)`.
    - Pass the resulting `RetrievalContext` into LLM prompt builders for
      reports and chat answers.
    """
    retrieved: List[RetrievedTransaction] = []

    for match in matches:
        meta = match.metadata or {}

        # Mirror the metadata keys used in fig.vector_store.TransactionDocument.
        transaction_id = meta.get("transaction_id") or match.id
        date = meta.get("date")
        amount = meta.get("amount")
        category = meta.get("category")
        product = meta.get("product")
        customer_id = meta.get("customer_id")
        channel = meta.get("channel")
        row_index = meta.get("row_index")

        retrieved.append(
            RetrievedTransaction(
                id=match.id,
                score=match.score,
                transaction_id=transaction_id,
                date=str(date) if date is not None else None,
                amount=float(amount) if isinstance(amount, (int, float)) else None,
                category=str(category) if category is not None else None,
                product=str(product) if product is not None else None,
                customer_id=str(customer_id) if customer_id is not None else None,
                channel=str(channel) if channel is not None else None,
                row_index=row_index,
                text=match.text,
                metadata=dict(meta),
            )
        )

    return RetrievalContext(
        query=query,
        matches=retrieved,
        top_k=top_k if top_k is not None else len(retrieved),
        language=language,
        filters=dict(filters) if filters is not None else {},
    )
"""
Embedding provider abstraction.

All embedding calls in the pipeline go through EmbeddingProvider.
No other file imports from fastembed, openai, or voyage directly.

To swap embedding models:
  1. Implement EmbeddingProvider (or use FastEmbedProvider with different model names)
  2. Re-run: python scripts/embed_catalog.py --rebuild
  Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class DenseVector:
    values: list[float]


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass
class EmbeddingPair:
    """One dense + one sparse vector for a single product or query."""
    dense: DenseVector
    sparse: SparseVector


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Interface for embedding providers.

    Implement this protocol to swap models without touching any other file.
    The retriever and embed_catalog.py use this interface exclusively.

    dense_text and sparse_text are passed separately because they are
    intentionally different (see Product.build_search_texts()). The provider
    does not decide what text to embed — the caller does.
    """

    @property
    def dense_dimensions(self) -> int:
        """
        Dimensionality of the dense vector.
        Must match the Qdrant collection schema — changing this requires --rebuild.
        """
        ...

    def embed_batch(
        self,
        dense_texts: list[str],
        sparse_texts: list[str],
    ) -> list[EmbeddingPair]:
        """
        Embed a batch of texts. dense_texts[i] and sparse_texts[i] are the
        two search representations for the same product or query.
        Returns one EmbeddingPair per input pair.
        """
        ...

    def embed_one(self, dense_text: str, sparse_text: str) -> EmbeddingPair:
        """Embed a single text pair. Convenience wrapper around embed_batch."""
        ...


class FastEmbedProvider:
    """
    Local CPU embedding via Qdrant's FastEmbed library.
    No API key. No network calls after initial model download (~100–500MB, cached).

    Default models:
      dense:  BAAI/bge-small-en-v1.5  (384 dims) — fast, good quality on CPU
      sparse: prithivida/Splade_PP_en_v1 — keyword-aware, exact term matching

    To use larger dense model (better quality, slower, needs --rebuild):
      FastEmbedProvider(dense_model="BAAI/bge-large-en-v1.5")
    """

    # Known dense model → output dimensions.
    # Add entries here when adding new dense models.
    _DENSE_DIMS: dict[str, int] = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }

    def __init__(
        self,
        dense_model: str = "BAAI/bge-small-en-v1.5",
        sparse_model: str = "prithivida/Splade_PP_en_v1",
    ) -> None:
        if dense_model not in self._DENSE_DIMS:
            raise ValueError(
                f"Unknown dense model '{dense_model}'. "
                f"Add it to FastEmbedProvider._DENSE_DIMS with its output dimensions."
            )

        # Lazy import — fastembed is optional until this class is instantiated.
        # Other parts of the codebase that don't do embedding don't pay the import cost.
        try:
            from fastembed import SparseTextEmbedding, TextEmbedding
        except ImportError as e:
            raise ImportError(
                "FastEmbed not installed. Run: pip install 'qdrant-client[fastembed]'"
            ) from e

        self._dense_model_name = dense_model
        self._sparse_model_name = sparse_model
        self._dense = TextEmbedding(model_name=dense_model)
        self._sparse = SparseTextEmbedding(model_name=sparse_model)

    @property
    def dense_dimensions(self) -> int:
        return self._DENSE_DIMS[self._dense_model_name]

    def embed_batch(
        self,
        dense_texts: list[str],
        sparse_texts: list[str],
    ) -> list[EmbeddingPair]:
        assert len(dense_texts) == len(sparse_texts), (
            "dense_texts and sparse_texts must have the same length"
        )

        dense_vecs = list(self._dense.embed(dense_texts))
        sparse_vecs = list(self._sparse.embed(sparse_texts))

        return [
            EmbeddingPair(
                dense=DenseVector(values=d.tolist()),
                sparse=SparseVector(
                    indices=s.indices.tolist(),
                    values=s.values.tolist(),
                ),
            )
            for d, s in zip(dense_vecs, sparse_vecs)
        ]

    def embed_one(self, dense_text: str, sparse_text: str) -> EmbeddingPair:
        return self.embed_batch([dense_text], [sparse_text])[0]


def default_provider() -> FastEmbedProvider:
    """
    Returns the default embedding provider using env-configured model names.
    Used by the retriever at query time and by embed_catalog.py at index time.
    """
    import os
    return FastEmbedProvider(
        dense_model=os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5"),
        sparse_model=os.getenv("SPARSE_MODEL", "prithivida/Splade_PP_en_v1"),
    )

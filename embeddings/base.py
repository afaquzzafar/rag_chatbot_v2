# ==============================================================================
# embeddings/base.py
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES
#   Defines EmbeddingProvider, an abstract interface every embedding backend
#   (Gemini, Databricks, or a future one) must implement.
#
# WHY AN ABSTRACT INTERFACE (LangChain / software design concept)
#   Every other module in this app (chunking->embedding, retrieval->embedding
#   the query) should be able to say "give me the vector for this text"
#   without caring *which* model produced it. That's the point of
#   programming to an interface instead of a concrete class: swapping
#   Gemini's hosted embedding model for Databricks' (or a future provider)
#   means changing ONE config value (`EMBEDDING_PROVIDER=databricks`), not
#   rewriting chunking, retrieval, or the vector store code.
#
# THE TWO METHODS, AND WHY THEY'RE SEPARATE
#   embed_documents(texts) -- embeds many chunks at once, at ingestion time.
#   embed_query(text)      -- embeds exactly one user question, at query time.
#   Some embedding models (including Gemini's) produce *different* vectors
#   for the same text depending on whether it's being stored as a document
#   or used as a search query (a "task type" hint that measurably improves
#   retrieval quality) -- so the interface bakes in that distinction from
#   the start rather than bolting it on later.
# ==============================================================================

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Abstract base class every embedding backend must implement."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of chunk texts for storage in the vector store.

        Args:
            texts: raw chunk strings.

        Returns:
            One embedding vector (list of floats) per input text, same order.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single user question for similarity search.

        Args:
            text: the user's question (already rewritten/standalone, if
                  query rewriting is enabled).

        Returns:
            One embedding vector (list of floats).
        """
        raise NotImplementedError

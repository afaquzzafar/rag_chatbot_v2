# ==============================================================================
# embeddings/embedding_service.py
# ------------------------------------------------------------------------------
# STEP 4 of the RAG pipeline: "Generate embeddings"
#
# WHAT THIS FILE DOES
#   The one place the rest of the app talks to in order to turn text into
#   vectors. It reads `settings.embedding_provider` and hands back the
#   matching EmbeddingProvider implementation (local or Gemini) -- this is
#   the "factory" pattern: callers ask for "an embedding provider" without
#   knowing or caring which concrete class they get back.
#
# WHY A SINGLETON FACTORY
#   Loading the local sentence-transformers model is slow (~1-2s, plus a
#   one-time download); creating a fresh Gemini client per call is wasteful.
#   `get_embedding_provider()` builds the provider once per process and
#   reuses it, the same way you wouldn't reconnect to a database on every
#   query.
#
# WHAT IS AN EMBEDDING? (GenAI beginners)
#   An embedding is a list of numbers (a "vector") that represents the
#   MEANING of a piece of text. The sentences
#       "What is my annual deductible?"
#       "How much do I pay before insurance starts covering costs?"
#   are worded completely differently but mean almost the same thing -- a
#   good embedding model places both vectors close together in a
#   high-dimensional mathematical space even though they share almost no
#   words. That's what makes RAG retrieval smarter than keyword search: it
#   finds the RIGHT chunk even when the question's wording doesn't match the
#   document's wording.
#
# INPUT / OUTPUT
#   generate_embeddings(texts) -> List[List[float]], one vector per text,
#   using whichever provider is currently configured.
# ==============================================================================

import threading
from typing import List, Optional

from config.settings import settings
from embeddings.base import EmbeddingProvider
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_provider_instance: Optional[EmbeddingProvider] = None
# Multi-query retrieval embeds several question variants concurrently
# (rag_pipeline.py's thread pool). Without this lock, the very first query
# could have every thread see `_provider_instance is None` and each load its
# own copy of the model at the same time (~420MB apiece for the local model).
_provider_lock = threading.Lock()


def get_embedding_provider() -> EmbeddingProvider:
    """
    Return the process-wide EmbeddingProvider singleton, building it on
    first call based on `settings.embedding_provider`. Thread-safe.
    """
    if _provider_instance is not None:
        return _provider_instance
    with _provider_lock:
        if _provider_instance is None:
            _build_provider()
    return _provider_instance


def _build_provider() -> None:
    """Construct the configured provider into `_provider_instance` (caller holds the lock)."""
    global _provider_instance

    if settings.embedding_provider == "gemini":
        from embeddings.gemini_embeddings import GeminiEmbeddingProvider

        logger.info("Using Gemini embeddings (model='%s')", settings.gemini_embedding_model)
        _provider_instance = GeminiEmbeddingProvider(
            model_name=settings.gemini_embedding_model,
            api_key=settings.gemini_api_key,
        )
    elif settings.embedding_provider == "databricks":
        from embeddings.databricks_embeddings import DatabricksEmbeddingProvider

        logger.info(
            "Using Databricks embeddings (endpoint='%s') -- keyless when run "
            "inside a Databricks workspace",
            settings.databricks_embedding_endpoint,
        )
        _provider_instance = DatabricksEmbeddingProvider(
            endpoint_name=settings.databricks_embedding_endpoint
        )
    elif settings.embedding_provider == "local":
        from embeddings.local_embeddings import LocalEmbeddingProvider

        logger.info("Using local embeddings (model='%s')", settings.local_embedding_model)
        _provider_instance = LocalEmbeddingProvider(model_name=settings.local_embedding_model)
    else:
        # settings.validate() should have already caught this, but a
        # defensive check here means this function never silently returns
        # None if it's ever called before validate().
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: '{settings.embedding_provider}'")


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using the currently configured provider.

    This is the reusable function every other module (chunking pipeline at
    ingest time, retrieval service at query time) should call -- it hides
    which concrete provider is active.

    Args:
        texts: list of chunk texts (or a single-item list for a query).

    Returns:
        One embedding vector per input text, same order.

    Raises:
        Whatever the underlying provider raises after exhausting its own
        retry policy (see embeddings/gemini_embeddings.py for the Gemini
        retry/backoff behavior). We deliberately do not swallow embedding
        failures here: a chunk that silently gets no vector would be
        invisible to search forever, which is worse than a loud failure at
        ingestion time.
    """
    if not texts:
        return []
    provider = get_embedding_provider()
    return provider.embed_documents(texts)


def generate_query_embedding(text: str) -> List[float]:
    """Embed a single user question using the currently configured provider."""
    provider = get_embedding_provider()
    return provider.embed_query(text)

# ==============================================================================
# tests/test_concurrent_init.py
# ------------------------------------------------------------------------------
# Multi-query retrieval (rag_pipeline/multi_query.py) searches several question
# variants at once on a thread pool. On the FIRST question of a process, every
# one of those threads hits the lazy "create once" singletons at the same
# moment. These tests pin down that those singletons are built exactly once
# and never fail under that concurrent first call (previously ChromaDB raised
# "Could not connect to tenant default_tenant", and the embedding model could
# be loaded once per thread).
# ==============================================================================

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from config.settings import settings
from embeddings.base import EmbeddingProvider

_THREADS = 8


def _run_concurrently(fn):
    """Call `fn` from _THREADS threads released at the same instant."""
    barrier = threading.Barrier(_THREADS)

    def call():
        barrier.wait()
        return fn()

    with ThreadPoolExecutor(max_workers=_THREADS) as executor:
        return list(executor.map(lambda _: call(), range(_THREADS)))


def test_chroma_collection_initializes_once_under_concurrent_first_use():
    import vector_store.chroma_manager as chroma_manager

    collections = _run_concurrently(chroma_manager._get_collection)

    assert all(c is collections[0] for c in collections)


def test_embedding_provider_is_built_once_under_concurrent_first_use(monkeypatch):
    import embeddings.embedding_service as embedding_service
    import embeddings.local_embeddings as local_embeddings

    constructed = []

    class SlowCountingProvider(EmbeddingProvider):
        def __init__(self, model_name: str):
            time.sleep(0.2)  # widen the race window, like a real model load
            constructed.append(model_name)

        def embed_documents(self, texts):
            return [[0.0] for _ in texts]

        def embed_query(self, text):
            return [0.0]

    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(local_embeddings, "LocalEmbeddingProvider", SlowCountingProvider)
    monkeypatch.setattr(embedding_service, "_provider_instance", None)

    providers = _run_concurrently(embedding_service.get_embedding_provider)

    assert len(constructed) == 1
    assert all(p is providers[0] for p in providers)

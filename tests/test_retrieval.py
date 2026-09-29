# ==============================================================================
# tests/test_retrieval.py
# ------------------------------------------------------------------------------
# Tests for rag_pipeline/retrieval_service.py and vector_store/chroma_manager.py.
#
# These run fully offline: the `isolated_settings` fixture (conftest.py)
# points ChromaDB at a temp directory and swaps in a deterministic
# FakeEmbeddingProvider, so no network call or real model is involved.
# We test *plumbing* here (top_k limiting, score-threshold filtering,
# metadata round-tripping) rather than semantic relevance, since the fake
# embeddings carry no real meaning.
# ==============================================================================

from chunking.chunker import Chunk
from embeddings.embedding_service import generate_embeddings
from rag_pipeline import retrieval_service
from vector_store import chroma_manager


def _seed_chunks(count: int = 6):
    chunks = [
        Chunk(
            chunk_id=f"chunk-{i}",
            chunk_text=f"This is sample chunk number {i} about plan benefits.",
            file_name="Summary_of_Benefits.pdf",
            page_number=i,
            document_type="Summary of Benefits",
        )
        for i in range(count)
    ]
    embeddings = generate_embeddings([c.chunk_text for c in chunks])
    chroma_manager.upsert_chunks(chunks, embeddings)
    return chunks


def test_distance_to_similarity_bounds():
    assert retrieval_service._distance_to_similarity(0.0) == 1.0
    assert retrieval_service._distance_to_similarity(2.0) == 0.0
    assert retrieval_service._distance_to_similarity(1.0) == 0.5
    # Out-of-range distances should clamp, not produce out-of-[0,1] scores.
    assert retrieval_service._distance_to_similarity(-1.0) == 1.0
    assert retrieval_service._distance_to_similarity(3.0) == 0.0


def test_retrieve_respects_top_k(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "enable_hybrid_search", False)
    _seed_chunks(count=6)

    results = retrieval_service.retrieve("plan benefits", top_k=3)

    assert len(results) == 3
    for chunk in results:
        assert chunk.document_name == "Summary_of_Benefits.pdf"
        assert chunk.document_type == "Summary of Benefits"
        assert 0.0 <= chunk.score <= 1.0


def test_retrieve_applies_score_threshold(monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "enable_hybrid_search", False)
    _seed_chunks(count=4)

    # An impossibly high threshold should filter out every result.
    results = retrieval_service.retrieve("plan benefits", top_k=10, score_threshold=1.1)
    assert results == []

    # A threshold of 0 should let everything through.
    results = retrieval_service.retrieve("plan benefits", top_k=10, score_threshold=0.0)
    assert len(results) == 4


def test_retrieve_on_empty_index_returns_empty_list():
    assert chroma_manager.count() == 0
    results = retrieval_service.retrieve("anything")
    assert results == []


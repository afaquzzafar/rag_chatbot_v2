# ==============================================================================
# conftest.py (project root)
# ------------------------------------------------------------------------------
# Shared pytest fixtures for the whole test suite.
#
# WHY THESE FIXTURES EXIST
#   The full app depends on three things we deliberately never want a unit
#   test to touch: a real Gemini API key, a real network call, and the
#   project's real on-disk vector store / SQLite file. These fixtures give
#   every test an isolated temp directory and a deterministic, offline fake
#   embedding provider, so `pytest` runs fully offline, fast, and without
#   ever mutating `vectorstore_db/` used by the real app.
# ==============================================================================

import hashlib
import sys
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings
from embeddings.base import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """
    A deterministic, offline stand-in for a real embedding model.

    Instead of a trained model, it hashes each text into a fixed-size
    vector. This is NOT semantically meaningful (it doesn't cluster similar
    meanings together) -- it exists purely so retrieval code can be tested
    for correct *plumbing* (does the right chunk_id/metadata come back,
    does score filtering work, etc.) without any network access or model
    download.
    """

    DIMENSIONS = 16

    def _hash_vector(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.DIMENSIONS]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_vector(text)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """
    Point every on-disk path this app uses at a fresh temp directory for
    every single test, and reset the process-wide singletons (embedding
    provider, Chroma client/collection) so tests never share state or touch
    the real project's vectorstore_db/.
    """
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "metadata_db_path", str(tmp_path / "metadata.sqlite"))
    monkeypatch.setattr(settings, "pdf_data_dir", str(tmp_path / "pdfs"))
    monkeypatch.setattr(settings, "mlflow_tracking_dir", str(tmp_path / "mlruns"))
    monkeypatch.setattr(settings, "feedback_log_path", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real")
    # The production default score_threshold (0.3) is meaningless against
    # FakeEmbeddingProvider's hash-based vectors, which carry no real
    # semantic similarity -- tests that expect a seeded chunk to be found
    # would fail or pass arbitrarily depending on hash luck. Tests that
    # specifically exercise threshold filtering pass an explicit value.
    monkeypatch.setattr(settings, "score_threshold", 0.0)
    # Gemini's real chat quota is only 5/minute; without this, a handful of
    # tests calling the (mocked) chat model in the same process would
    # actually sleep for real, since the rate limiter runs regardless of
    # whether the underlying LLM call itself is mocked.
    monkeypatch.setattr(settings, "gemini_chat_requests_per_minute", 100_000)

    import vector_store.chroma_manager as chroma_manager

    monkeypatch.setattr(chroma_manager, "_client", None)
    monkeypatch.setattr(chroma_manager, "_collection", None)

    import embeddings.embedding_service as embedding_service

    monkeypatch.setattr(embedding_service, "_provider_instance", FakeEmbeddingProvider())

    import rag_pipeline.llm_service as llm_service

    monkeypatch.setattr(llm_service, "_llm_instance", None)
    monkeypatch.setattr(llm_service, "_chat_rate_limiter", None)

    yield


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    """
    Build a small, real, synthetic PDF (via PyMuPDF) with known text on
    each page -- lets ingestion/chunking tests run against a genuine PDF
    file without shipping a real insurance document into the test suite.
    """
    pdf_path = tmp_path / "Summary_of_Benefits.pdf"
    doc = fitz.open()

    page_texts = [
        "Annual Deductible\n\nThe annual deductible for this plan is $250 per member "
        "per calendar year. This amount must be paid before the plan begins to share "
        "the cost of covered services.",
        "Dental Coverage\n\nThis plan includes preventive dental coverage, including "
        "two cleanings per year at no additional cost to the member.",
    ]
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)

    doc.save(pdf_path)
    doc.close()
    return pdf_path

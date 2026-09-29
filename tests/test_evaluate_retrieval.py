# ==============================================================================
# tests/test_evaluate_retrieval.py
# ------------------------------------------------------------------------------
# Tests for scripts/evaluate_retrieval.py's pure helper functions: locating
# the correct page's rank, computing one question's metrics, and aggregating
# across queries. The live RAGPipeline.retrieve() call itself is covered in
# tests/test_rag_pipeline.py -- these tests are about the math on top of it.
# ==============================================================================

import pytest

from rag_pipeline.retrieval_service import RetrievedChunk
from scripts.eval_questions import EvalQuestion
from scripts.evaluate_retrieval import (
    StageMetrics,
    _aggregate_stage,
    _corpus_fingerprint,
    _find_rank,
    _stage_metrics,
)


def _chunk(doc: str, page: int, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(chunk_text="text", score=score, document_name=doc, document_type="Test", page_number=page)


def test_find_rank_locates_the_correct_page():
    eval_q = EvalQuestion("Q", "doc.pdf", 3)
    chunks = [_chunk("doc.pdf", 1), _chunk("doc.pdf", 3), _chunk("doc.pdf", 5)]
    assert _find_rank(chunks, eval_q) == 2


def test_find_rank_returns_none_when_the_page_never_appears():
    eval_q = EvalQuestion("Q", "doc.pdf", 99)
    assert _find_rank([_chunk("doc.pdf", 1)], eval_q) is None


def test_stage_metrics_are_perfect_when_found_at_rank_one():
    eval_q = EvalQuestion("Q", "doc.pdf", 1)
    metrics = _stage_metrics([_chunk("doc.pdf", 1)], eval_q)

    assert metrics.found_rank == 1
    assert metrics.recall_at_k[1] == 1.0
    assert metrics.ndcg_at_k[1] == 1.0
    assert metrics.reciprocal_rank == 1.0


def test_stage_metrics_are_zero_when_never_found():
    eval_q = EvalQuestion("Q", "doc.pdf", 99)
    metrics = _stage_metrics([_chunk("doc.pdf", 1)], eval_q)

    assert metrics.found_rank is None
    assert metrics.recall_at_k[5] == 0.0
    assert metrics.reciprocal_rank == 0.0


def test_aggregate_stage_averages_metrics_across_queries():
    perfect = StageMetrics(1, {1: 1.0, 3: 1.0, 5: 1.0}, {1: 1.0, 3: 1 / 3, 5: 0.2}, {1: 1.0, 3: 1.0, 5: 1.0}, 1.0)
    missed = StageMetrics(None, {1: 0.0, 3: 0.0, 5: 0.0}, {1: 0.0, 3: 0.0, 5: 0.0}, {1: 0.0, 3: 0.0, 5: 0.0}, 0.0)

    aggregate = _aggregate_stage([perfect, missed])

    assert aggregate["recall_at_k"][1] == pytest.approx(0.5)
    assert aggregate["mrr"] == pytest.approx(0.5)


def test_corpus_fingerprint_changes_when_a_pdf_changes(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "a.pdf").write_bytes(b"version one")
    first = _corpus_fingerprint(str(pdf_dir))

    (pdf_dir / "a.pdf").write_bytes(b"version two")
    second = _corpus_fingerprint(str(pdf_dir))

    assert first != second


def test_corpus_fingerprint_is_stable_when_nothing_changed(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "a.pdf").write_bytes(b"content")

    assert _corpus_fingerprint(str(pdf_dir)) == _corpus_fingerprint(str(pdf_dir))


def test_corpus_fingerprint_is_empty_for_a_missing_directory(tmp_path):
    assert _corpus_fingerprint(str(tmp_path / "does_not_exist")) == ""

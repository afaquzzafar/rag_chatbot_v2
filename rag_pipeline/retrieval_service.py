# ==============================================================================
# rag_pipeline/retrieval_service.py
# ------------------------------------------------------------------------------
# STEP: "Retrieve Top K Chunks" (+ ENHANCEMENT: "Hybrid Search")
#
# WHAT THIS FILE DOES
#   Given a question, returns the top_k most relevant chunks from the
#   vector store, each with a similarity score and full source metadata
#   (document name, page number) ready for citation.
#
# WHY A SEPARATE "SERVICE" CLASS INSTEAD OF CALLING chroma_manager DIRECTLY
#   RAGPipeline (rag_pipeline.py) shouldn't need to know HOW retrieval works
#   (vector-only vs. hybrid) -- only that it
#   can call `retrieve(question, top_k)` and get back ranked chunks. This
#   class is the seam where retrieval strategy can evolve (e.g. add hybrid
#   search, tune the threshold) without touching the pipeline that calls it.
#
# WHY HYBRID SEARCH (vector + BM25 keyword search)
#   Vector search is excellent at matching MEANING ("how much do I pay
#   before coverage starts" ~ "deductible") but can occasionally miss exact,
#   rare tokens that matter a lot in insurance documents -- plan codes,
#   specific dollar amounts, drug names, CPT codes. BM25 is a classic
#   keyword-frequency algorithm that's excellent at exactly that: exact
#   term matches. Combining both (weighted sum of normalized scores) covers
#   each method's blind spot with the other's strength.
#
# SIMILARITY SCORE: converting Chroma's "distance" to an intuitive score
#   Chroma's cosine-space collection returns a *distance* (0 = identical,
#   2 = opposite). We convert to a 0-1 "similarity" score via
#   `score = 1 - (distance / 2)` so the rest of the app (and the UI's
#   "confidence" display) can work with the more intuitive "higher is
#   better, ~1.0 is a great match" convention.
#
# INPUT / OUTPUT
#   retrieve(question, top_k, score_threshold) -> List[RetrievedChunk]
# ==============================================================================

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config.settings import settings
from embeddings.embedding_service import generate_query_embedding
from utils.logging_utils import get_logger
from vector_store import chroma_manager

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """One retrieved chunk plus everything the UI/prompt needs to use it."""

    chunk_text: str
    score: float  # 0.0-1.0, higher = more relevant
    document_name: str
    document_type: str
    page_number: int
    # Populated only for chapter/section-structured documents (see
    # chunking/structure_chunker.py); "" for flatter documents.
    chapter_title: str = ""
    section_title: str = ""


def chunk_identity(chunk: RetrievedChunk) -> tuple:
    """
    A stable de-duplication key for a retrieved chunk.

    RetrievedChunk doesn't carry the vector store's own chunk_id, so
    (document, page, text) stands in for it -- used by multi_query.py's
    fusion and rag_pipeline.py's multi-hop merging to recognize the "same"
    chunk surfaced by two different queries.
    """
    return (chunk.document_name, chunk.page_number, chunk.chunk_text)


def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma's cosine distance (0=identical, 2=opposite) to a 0-1 score."""
    similarity = 1.0 - (distance / 2.0)
    return max(0.0, min(1.0, similarity))


def _vector_search(query: str, top_k: int) -> List[RetrievedChunk]:
    """Run pure vector similarity search and return candidate chunks."""
    query_embedding = generate_query_embedding(query)
    raw = chroma_manager.similarity_search(query_embedding, top_k=top_k)

    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    results: List[RetrievedChunk] = []
    for text, meta, distance in zip(documents, metadatas, distances):
        results.append(
            RetrievedChunk(
                chunk_text=text,
                score=_distance_to_similarity(distance),
                document_name=meta.get("document_name", "unknown"),
                document_type=meta.get("document_type", "unknown"),
                page_number=meta.get("page_number", 0),
                chapter_title=meta.get("chapter_title", ""),
                section_title=meta.get("section_title", ""),
            )
        )
    return results


def _hybrid_rescore(query: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """
    Blend vector similarity with a BM25 keyword score computed over just the
    candidate set (not the whole corpus -- BM25 here is a fast local
    re-scoring step, not a separate full-corpus retrieval pass).

    Blend weights (70% vector / 30% keyword) favor semantic matching, since
    that's the stronger signal for natural-language insurance questions,
    while still rewarding exact keyword hits (plan codes, dollar figures).
    """
    from rank_bm25 import BM25Okapi

    if not candidates:
        return candidates

    tokenized_corpus = [c.chunk_text.lower().split() for c in candidates]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())

    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    for candidate, raw_bm25 in zip(candidates, bm25_scores):
        normalized_bm25 = raw_bm25 / max_bm25
        candidate.score = (0.7 * candidate.score) + (0.3 * normalized_bm25)

    return sorted(candidates, key=lambda c: c.score, reverse=True)


def retrieve(
    question: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> List[RetrievedChunk]:
    """
    Retrieve the relevant candidate chunks for a question.

    Args:
        question: the (rewritten, standalone) user question.
        top_k: how many chunks to return; defaults to settings.top_k.
        score_threshold: drop chunks scoring below this; defaults to
                         settings.score_threshold (0.0 = no filtering).

    Returns:
        Up to top_k RetrievedChunk objects, best-first.
    """
    top_k = top_k if top_k is not None else settings.top_k
    score_threshold = score_threshold if score_threshold is not None else settings.score_threshold

    # Over-fetch a slightly larger candidate pool than top_k when hybrid
    # search will run afterward, so it has enough candidates to meaningfully
    # re-sort rather than just re-sorting an already-truncated list of
    # `top_k` items.
    fetch_k = top_k * 3 if settings.enable_hybrid_search else top_k

    candidates = _vector_search(question, top_k=fetch_k)

    if settings.enable_hybrid_search:
        candidates = _hybrid_rescore(question, candidates)

    filtered = [c for c in candidates if c.score >= score_threshold]

    results = filtered[:top_k]

    logger.info(
        "Retrieved %d chunks for question (top_k=%d, threshold=%.2f, hybrid=%s)",
        len(results),
        top_k,
        score_threshold,
        settings.enable_hybrid_search,
    )
    return results


def retrieve_as_dicts(
    question: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Same as retrieve(), but returns plain dicts -- the shape
    MLflow logging expects.
    """
    chunks = retrieve(question, top_k=top_k, score_threshold=score_threshold)
    return [
        {
            "chunk_text": c.chunk_text,
            "score": c.score,
            "document_name": c.document_name,
            "document_type": c.document_type,
            "page_number": c.page_number,
            "chapter_title": c.chapter_title,
            "section_title": c.section_title,
        }
        for c in chunks
    ]

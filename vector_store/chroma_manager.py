# ==============================================================================
# vector_store/chroma_manager.py
# ------------------------------------------------------------------------------
# STEP 6-7 of the RAG pipeline: "Store Embeddings" + "Create Vector Search
# Index" -- implemented locally with ChromaDB instead of Databricks Vector
# Search.
#
# WHAT THIS FILE DOES
#   Wraps a ChromaDB "collection" (Chroma's name for a vector index) so the
#   rest of the app can upsert chunks and run similarity search without
#   touching the ChromaDB API directly.
#
# WHAT IS A VECTOR INDEX, AND WHY CHROMA HERE
#   A vector index is a data structure (typically HNSW -- Hierarchical
#   Navigable Small World graphs) built specifically to answer "which of my
#   millions of stored vectors are closest to this new vector?" in
#   milliseconds, instead of comparing against every single vector one by
#   one (which is what a naive linear scan does). ChromaDB is an embedded,
#   on-disk vector database: `chromadb.PersistentClient(path=...)` stores
#   everything in a local folder, with no separate server process to run --
#   ideal for a local sandbox / single-machine deployment.
#
# DATABRICKS CONCEPT MAPPING
#   In production on Databricks, this same role is played by a Databricks
#   Vector Search *endpoint* (the compute serving the index) plus a Vector
#   Search *index* synced from a Delta table (the "Create Databricks Vector
#   Search Index" + "Index synchronization" steps from the original spec).
#   The concepts map directly:
#     Chroma "collection"          <-> Databricks Vector Search "index"
#     collection.upsert(...)       <-> Delta table write + index sync
#     collection.query(...)        <-> vs_index.similarity_search(...)
#   Swapping this module for a Databricks-backed one later means the rest of
#   the app (retrieval_service.py) does not need to change, because it only
#   ever calls this module's functions, never the ChromaDB client directly.
#
# WHY "cosine" DISTANCE
#   We configure the collection's `hnsw:space` metadata to "cosine" so
#   similarity is measured by the *angle* between vectors rather than their
#   raw magnitude -- the standard choice for text embeddings, where two
#   chunks meaning the same thing should be "close" regardless of length.
#
# INPUT / OUTPUT
#   upsert_chunks(chunks, embeddings)               -> None
#   similarity_search(query_embedding, top_k, where) -> raw Chroma results dict
#   count() -> int (used to detect "collection is empty, run ingestion")
# ==============================================================================

import threading
from typing import Any, Dict, List, Optional

import chromadb

from chunking.chunker import Chunk
from config.settings import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[Any] = None
# Multi-query retrieval searches several question variants concurrently
# (rag_pipeline.py's thread pool). Opening PersistentClient from several
# threads at once fails with "Could not connect to tenant default_tenant",
# so first-time initialization is serialized.
_init_lock = threading.Lock()


def _get_collection():
    """
    Lazily create (once per process) the persistent Chroma client and
    collection, matching the singleton pattern used elsewhere in this app
    to avoid re-opening the on-disk database on every call. Thread-safe.
    """
    global _client, _collection
    if _collection is not None:
        return _collection

    with _init_lock:
        if _collection is not None:
            return _collection

        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready at '%s' (%d chunks currently stored)",
            settings.chroma_collection_name,
            settings.chroma_persist_dir,
            _collection.count(),
        )
    return _collection


def upsert_chunks(chunks: List[Chunk], embeddings: List[List[float]]) -> None:
    """
    Insert or update chunks (and their embeddings) in the vector index.

    "Upsert" (update-or-insert) means re-ingesting the same chunk_id twice
    (e.g. after re-processing a changed PDF) overwrites the old vector
    instead of creating a duplicate entry -- important for the idempotent,
    hash-based re-ingestion logic in scripts/ingest.py.
    """
    if not chunks:
        return
    collection = _get_collection()
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.chunk_text for c in chunks],
        metadatas=[
            {
                "document_name": c.file_name,
                "document_type": c.document_type,
                "page_number": c.page_number,
                "created_timestamp": c.created_timestamp,
                # Chroma metadata values can't be None -- "" means "not a
                # chapter/section-structured document" (see chunking/chunker.py).
                "chapter_title": c.chapter_title or "",
                "section_title": c.section_title or "",
            }
            for c in chunks
        ],
    )
    logger.info("Upserted %d chunks into the vector index", len(chunks))


def delete_by_document(document_name: str) -> None:
    """
    Remove every chunk belonging to one source document.

    Used when a PDF changes (different file hash) so its stale chunks are
    removed before the freshly re-chunked, re-embedded version is inserted --
    otherwise the index would accumulate both old and new chunks forever.
    """
    collection = _get_collection()
    collection.delete(where={"document_name": document_name})
    logger.info("Deleted existing chunks for document '%s'", document_name)


def similarity_search(
    query_embedding: List[float],
    top_k: int,
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Any]]:
    """
    Run a top-K similarity search against the vector index.

    Args:
        query_embedding: the embedded user question.
        top_k: how many nearest chunks to return.
        where: optional Chroma metadata filter, e.g. {"document_type": "Summary of Benefits"}.

    Returns:
        Chroma's raw query result: a dict of parallel lists
        (ids, documents, metadatas, distances), each wrapped one extra list
        deep because Chroma supports batched queries -- callers should read
        result["ids"][0], result["documents"][0], etc.
    """
    collection = _get_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )


def count() -> int:
    """Return how many chunks are currently stored -- used to detect an empty index."""
    return _get_collection().count()

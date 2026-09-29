# ==============================================================================
# chunking/chunker.py
# ------------------------------------------------------------------------------
# STEP 3 of the RAG pipeline: "Chunk Documents"
#
# WHAT THIS FILE DOES
#   Splits each page's text into smaller overlapping "chunks" of ~1000
#   characters, and attaches metadata (chunk_id, file_name, page_number,
#   document_type) to every chunk so it can later be cited and traced back
#   to its exact source page.
#
# WHY CHUNK AT ALL? (GenAI beginners)
#   An LLM's context window is finite, and even within that limit, stuffing
#   an entire 100-page policy document into every prompt would be slow,
#   expensive, and would bury the relevant sentence in a haystack of
#   irrelevant text. Instead, we split documents into small pieces ahead of
#   time, embed each piece separately, and at question time retrieve only
#   the handful of pieces that are actually relevant to the question.
#
# WHY RecursiveCharacterTextSplitter, SPECIFICALLY
#   LangChain's RecursiveCharacterTextSplitter tries to split on the most
#   "natural" boundary first (paragraph breaks "\n\n"), and only falls back
#   to less natural boundaries (single newlines, sentences, words, then raw
#   characters) if a piece is still too long. This matters a lot for
#   insurance documents: it strongly prefers to keep a whole paragraph (e.g.
#   one complete benefit description) together in one chunk, rather than
#   cutting it in half at an arbitrary character count the way a naive
#   fixed-size splitter would.
#
# WHY SOME DOCUMENTS SKIP THIS AND USE A HIERARCHICAL SPLITTER INSTEAD
#   The Evidence of Coverage has real chapters and numbered sections (see
#   chunking/structure_chunker.py). For that document, splitting purely on
#   character count would ignore its actual structure and could carve a
#   single benefit explanation into two unrelated chunks. chunk_documents()
#   detects that structure per-file and, when present, splits along chapter/
#   section boundaries first -- falling back to this same recursive splitter
#   only for the rare section still too large to embed as one chunk. The
#   other, flatter documents (Summary of Benefits, Policy Manual) have no
#   such structure to exploit and keep using plain recursive splitting.
#
# WHY CHUNK_SIZE=1000 / CHUNK_OVERLAP=200
#   - 1000 characters (~150-200 words) is large enough to contain a complete
#     thought (e.g. "Annual deductible: $250 per member...") but small
#     enough that retrieving 5 of them (top_k=5) still fits comfortably in
#     an LLM prompt alongside the question and system instructions.
#   - 200 characters of overlap (20% of chunk_size) protects against the
#     single worst failure mode of chunking: a key sentence landing exactly
#     on a chunk boundary and getting split in half, becoming useless in
#     BOTH resulting chunks. The overlap means content near a boundary
#     appears fully intact in at least one chunk.
#
# INPUT / OUTPUT
#   Input:  List[PageContent] from ingestion/pdf_loader.py.
#   Output: List[Chunk], each with unique chunk_id + full source metadata.
# ==============================================================================

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import groupby
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.structure_chunker import looks_structured, split_document
from config.settings import settings
from ingestion.pdf_loader import PageContent
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A single chunk of text plus everything needed to store, embed, and cite it."""

    chunk_id: str
    chunk_text: str
    file_name: str
    page_number: int
    document_type: str
    # Only set for documents split by chunking/structure_chunker.py (e.g. the
    # Evidence of Coverage's chapters/sections); None for flatter documents
    # chunked by plain recursive splitting.
    chapter_title: Optional[str] = None
    section_title: Optional[str] = None
    created_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _chunk_file_recursively(pages: List[PageContent]) -> List[Chunk]:
    """Plain size-based chunking, per page -- used for unstructured documents."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Try paragraph breaks first, then lines, then sentences, then words,
        # then raw characters -- in that priority order.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Chunk] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    chunk_text=piece,
                    file_name=page.file_name,
                    page_number=page.page_number,
                    document_type=page.document_type,
                )
            )
    return chunks


def _chunk_file_hierarchically(pages: List[PageContent]) -> List[Chunk]:
    """Chapter/section-aware chunking -- used for documents with real structure."""
    pieces = split_document(pages, settings.chunk_size, settings.chunk_overlap)

    chunks: List[Chunk] = []
    for piece in pieces:
        # A short "breadcrumb" of where this chunk sits in the document,
        # prepended to its text. This keeps a chunk meaningful even once it's
        # retrieved on its own, out of context (the embedding model sees it
        # too), and mirrors the citation shown to users.
        breadcrumb_parts = [p for p in (piece.chapter_title, piece.section_title) if p]
        chunk_text = f"{' > '.join(breadcrumb_parts)}\n{piece.text}" if breadcrumb_parts else piece.text

        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                chunk_text=chunk_text,
                file_name=pages[0].file_name,
                page_number=piece.page_number,
                document_type=pages[0].document_type,
                chapter_title=piece.chapter_title,
                section_title=piece.section_title,
            )
        )
    return chunks


def chunk_documents(pages: List[PageContent]) -> List[Chunk]:
    """
    Split page-level texts into overlapping chunks with metadata.

    Documents are chunked one file at a time (grouped by file_name, assuming
    ingestion.pdf_loader.load_pdfs()'s per-file page ordering is preserved):
    a document with real chapter/section headings is split hierarchically
    (chunking/structure_chunker.py); everything else uses plain recursive,
    size-based splitting.

    Args:
        pages: output of ingestion.pdf_loader.load_pdfs().

    Returns:
        A flat list of Chunk objects across all input pages.
    """
    chunks: List[Chunk] = []
    for file_name, file_pages_iter in groupby(pages, key=lambda p: p.file_name):
        file_pages = list(file_pages_iter)
        if looks_structured(file_pages):
            logger.info("'%s' has chapter/section structure -- using hierarchical chunking", file_name)
            chunks.extend(_chunk_file_hierarchically(file_pages))
        else:
            chunks.extend(_chunk_file_recursively(file_pages))

    logger.info(
        "Chunked %d pages into %d chunks (chunk_size=%d, chunk_overlap=%d)",
        len(pages),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks

# ==============================================================================
# embeddings/gemini_embeddings.py
# ------------------------------------------------------------------------------
# DEFAULT PROVIDER: calls the Gemini API over HTTPS -- no model weights are
# downloaded, so it also works on networks that block huggingface.co.
#
# WHAT THIS FILE DOES
#   Implements EmbeddingProvider using Google's hosted Gemini embedding
#   model (default: "models/gemini-embedding-001") via the `google-generativeai`
#   SDK -- the same GEMINI_API_KEY already used for chat answers.
#
# WHY A SEPARATE "task_type" FOR DOCUMENTS VS. QUERIES
#   Gemini's embedding API accepts a `task_type` hint: "RETRIEVAL_DOCUMENT"
#   when embedding text that will be *stored and searched*, and
#   "RETRIEVAL_QUERY" when embedding the *question* being asked. Internally
#   the model can produce asymmetric embeddings optimized for
#   query-to-document matching rather than plain document-to-document
#   similarity -- using the right hint measurably improves retrieval
#   accuracy over using one generic embedding for both.
#
# BATCHING + RETRY (why both are needed for a production-grade service)
#   - Batching: embedding 500 chunks one HTTP call at a time would be slow
#     and wasteful; we send them in configurable-size batches instead
#     (settings.embedding_batch_size).
#   - Retry (`tenacity`): network calls fail transiently (rate limits,
#     brief outages). Retrying a handful of times with exponential backoff
#     turns a flaky blip into a successful call instead of a crashed
#     ingestion run.
#
# INPUT / OUTPUT
#   Input:  list of strings (documents) or one string (query).
#   Output: list of float vectors (or one vector for embed_query).
# ==============================================================================

from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from embeddings.base import EmbeddingProvider
from utils.logging_utils import get_logger
from utils.rate_limiter import SlidingWindowRateLimiter

logger = get_logger(__name__)


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Hosted embeddings via the Gemini API -- no local model download required."""

    def __init__(self, model_name: str, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model_name
        self._rate_limiter = SlidingWindowRateLimiter(
            settings.gemini_embedding_requests_per_minute, name="Gemini embedding requests"
        )
        logger.info("Gemini embedding provider ready (model='%s')", model_name)

    @retry(
        # The API's own 429 responses recommend waiting ~50s before retrying,
        # so this backoff needs real headroom -- not just a quick blip retry.
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=70),
        reraise=True,
    )
    def _embed_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        """
        Embed one batch of texts, first pacing against the rate limiter,
        then retrying transient failures (including quota 429s) with
        exponential backoff.
        """
        self._rate_limiter.acquire(len(texts))
        try:
            response = self._genai.embed_content(
                model=self._model_name,
                content=texts,
                task_type=task_type,
            )
        except Exception:
            logger.exception(
                "Gemini embed_content failed for a batch of %d texts (task_type=%s)",
                len(texts),
                task_type,
            )
            raise
        return response["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = settings.embedding_batch_size
        all_vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = self._embed_batch(batch, task_type="RETRIEVAL_DOCUMENT")
            all_vectors.extend(vectors)
            logger.debug(
                "Embedded documents %d-%d of %d", start, start + len(batch), len(texts)
            )
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embed_batch([text], task_type="RETRIEVAL_QUERY")
        return vectors[0]
